#!/usr/bin/env python3
"""
Main execution script for the Network Performance AI Anomaly Detection System.

This script implements the complete monitoring lifecycle:
1. Connect to database and get node information (KPIs)
2. Connect to ticketing system and get existing tickets information
3. Apply detection algorithm on nodes that don't have TT (trouble tickets)
4. Raise TT for detected issues
5. Send email for detected issues including charts and TT number

Usage:
    python scripts/run_monitoring.py [--config CONFIG_PATH] [--kpi KPI_NAME] [--node NODE_NAME]
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import get_config
from src.utils.logger import get_monitoring_logger, log_performance
from src.data.database import get_database
from src.data.models import KPIQuery, SystemMetrics
from src.services.monitoring import NetworkMonitoringService
from src.services.ticketing import TicketingService
from src.services.email_service import EmailService
from src.core.detector import AnomalyDetector


class MonitoringOrchestrator:
    """
    Main orchestrator for the network monitoring system.
    
    This class coordinates all components to implement the complete
    monitoring lifecycle as specified.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the monitoring orchestrator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config(config_path)
        self.logger = get_monitoring_logger()
        
        # Initialize services
        self.database = get_database(self.config)
        self.monitoring_service = NetworkMonitoringService(self.config)
        self.ticketing_service = TicketingService(self.config)
        self.email_service = EmailService(self.config)
        self.detector = AnomalyDetector(self.config)
        
        # Metrics tracking
        self.metrics = SystemMetrics(
            timestamp=datetime.now(),
            total_nodes_monitored=0,
            kpis_processed=0,
            anomalies_detected=0,
            tickets_raised=0,
            emails_sent=0,
            processing_time_seconds=0.0,
            errors_count=0
        )
        
        self.logger.info("Monitoring orchestrator initialized")
    
    @log_performance
    def run_complete_cycle(
        self,
        kpi_filter: Optional[str] = None,
        node_filter: Optional[str] = None
    ) -> SystemMetrics:
        """
        Run the complete monitoring cycle.
        
        Args:
            kpi_filter: Specific KPI to monitor (optional)
            node_filter: Specific node to monitor (optional)
            
        Returns:
            System metrics from the monitoring cycle
        """
        start_time = time.time()
        self.logger.info("Starting complete monitoring cycle")
        
        try:
            # Step 1: Connect to database and get node information
            self.logger.info("Step 1: Fetching node information and KPI data")
            kpi_results = self._fetch_kpi_data(kpi_filter, node_filter)
            
            # Step 2: Connect to ticketing system and get existing tickets
            self.logger.info("Step 2: Fetching existing ticket information")
            existing_tickets = self._get_existing_tickets()
            
            # Step 3: Apply detection algorithm on nodes without tickets
            self.logger.info("Step 3: Running anomaly detection")
            detection_results = self._run_anomaly_detection(kpi_results, existing_tickets)
            
            # Step 4: Raise tickets for detected issues
            self.logger.info("Step 4: Raising tickets for detected anomalies")
            raised_tickets = self._raise_tickets(detection_results)
            
            # Step 5: Send email notifications
            self.logger.info("Step 5: Sending email notifications")
            email_results = self._send_notifications(detection_results, raised_tickets)
            
            # Update metrics
            self.metrics.processing_time_seconds = time.time() - start_time
            self.metrics.tickets_raised = len(raised_tickets)
            self.metrics.emails_sent = len(email_results)
            
            self.logger.info(
                f"Monitoring cycle completed successfully. "
                f"Processed {self.metrics.kpis_processed} KPIs, "
                f"detected {self.metrics.anomalies_detected} anomalies, "
                f"raised {self.metrics.tickets_raised} tickets, "
                f"sent {self.metrics.emails_sent} emails"
            )
            
            return self.metrics
            
        except Exception as e:
            self.metrics.errors_count += 1
            self.metrics.processing_time_seconds = time.time() - start_time
            self.logger.error(f"Monitoring cycle failed: {e}", exc_info=True)
            raise
    
    def _fetch_kpi_data(
        self,
        kpi_filter: Optional[str] = None,
        node_filter: Optional[str] = None
    ) -> dict:
        """
        Fetch KPI data for all enabled KPIs.
        
        Args:
            kpi_filter: Specific KPI to fetch (optional)
            node_filter: Specific node to fetch (optional)
            
        Returns:
            Dictionary with KPI results by node and KPI type
        """
        kpi_results = {}
        
        # Get enabled KPIs
        enabled_kpis = self.config.get_enabled_kpis()
        if kpi_filter:
            enabled_kpis = [kpi for kpi in enabled_kpis if kpi.name == kpi_filter]
        
        # Get active nodes if no specific node filter
        if node_filter:
            active_nodes = [node_filter]
        else:
            active_nodes = self.database.get_active_nodes(
                hours_back=self.config.monitoring.data_collection.get('default_range_hours', 168)
            )
        
        self.metrics.total_nodes_monitored = len(active_nodes)
        
        # Time range for data collection
        end_time = datetime.now()
        start_time = end_time - timedelta(
            hours=self.config.monitoring.data_collection.get('default_range_hours', 168)
        )
        
        for kpi_config in enabled_kpis:
            self.logger.info(f"Fetching data for KPI: {kpi_config.name}")
            
            for node in active_nodes:
                try:
                    # Create query
                    query = KPIQuery(
                        kpi_name=kpi_config.name,
                        start_time=start_time,
                        end_time=end_time,
                        snode_filter=node
                    )
                    
                    # Fetch data
                    kpi_data = self.database.fetch_kpi_data(query)
                    
                    if kpi_data:
                        if node not in kpi_results:
                            kpi_results[node] = {}
                        kpi_results[node][kpi_config.name] = {
                            'data': kpi_data,
                            'config': kpi_config
                        }
                        self.metrics.kpis_processed += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch KPI data for {node}/{kpi_config.name}: {e}")
                    self.metrics.errors_count += 1
        
        self.logger.info(f"Fetched data for {self.metrics.kpis_processed} KPI/node combinations")
        return kpi_results
    
    def _get_existing_tickets(self) -> dict:
        """
        Get existing tickets from the ticketing system.
        
        Returns:
            Dictionary with existing tickets by node
        """
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            existing_tickets = self.ticketing_service.query_tickets(current_date)
            
            # Organize tickets by node
            tickets_by_node = {}
            for ticket in existing_tickets:
                node = ticket.get('description', '').split(' ')[0]  # Assuming node is first word
                if node not in tickets_by_node:
                    tickets_by_node[node] = []
                tickets_by_node[node].append(ticket)
            
            self.logger.info(f"Found {len(existing_tickets)} existing tickets")
            return tickets_by_node
            
        except Exception as e:
            self.logger.error(f"Failed to get existing tickets: {e}")
            return {}
    
    def _run_anomaly_detection(self, kpi_results: dict, existing_tickets: dict) -> list:
        """
        Run anomaly detection on nodes without existing tickets.
        
        Args:
            kpi_results: KPI data by node and type
            existing_tickets: Existing tickets by node
            
        Returns:
            List of detection results with anomalies
        """
        detection_results = []
        
        for node, kpi_data in kpi_results.items():
            # Skip nodes with existing tickets (unless forced)
            if node in existing_tickets:
                self.logger.info(f"Skipping {node} - existing ticket found")
                continue
            
            for kpi_name, kpi_info in kpi_data.items():
                try:
                    self.logger.info(f"Running detection for {node}/{kpi_name}")
                    
                    # Get detection algorithm configuration
                    algorithm = kpi_info['config'].detection_algorithm
                    threshold_config = kpi_info['config'].threshold_config
                    
                    # Run detection
                    result = self.detector.detect_anomalies(
                        kpi_data=kpi_info['data'],
                        algorithm=algorithm,
                        threshold_config=threshold_config,
                        snode=node,
                        kpi_name=kpi_name
                    )
                    
                    if result.has_anomalies:
                        detection_results.append(result)
                        self.metrics.anomalies_detected += len(result.anomalies)
                        self.logger.info(
                            f"Detected {len(result.anomalies)} anomalies for {node}/{kpi_name}"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Detection failed for {node}/{kpi_name}: {e}")
                    self.metrics.errors_count += 1
        
        self.logger.info(f"Detection completed: {len(detection_results)} nodes with anomalies")
        return detection_results
    
    def _raise_tickets(self, detection_results: list) -> list:
        """
        Raise tickets for detected anomalies.
        
        Args:
            detection_results: List of detection results with anomalies
            
        Returns:
            List of successfully raised tickets
        """
        raised_tickets = []
        
        for result in detection_results:
            try:
                # Get node location
                location = self.database.get_node_location(result.snode)
                
                # Create ticket
                ticket_info = self.ticketing_service.raise_ticket(
                    snode=result.snode,
                    kpi_name=result.kpi_name,
                    detection_result=result,
                    location=location
                )
                
                if ticket_info and ticket_info.ticket_id:
                    raised_tickets.append(ticket_info)
                    self.logger.info(
                        f"Raised ticket {ticket_info.ticket_id} for {result.snode}/{result.kpi_name}"
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to raise ticket for {result.snode}: {e}")
                self.metrics.errors_count += 1
        
        return raised_tickets
    
    def _send_notifications(self, detection_results: list, raised_tickets: list) -> list:
        """
        Send email notifications for detected anomalies.
        
        Args:
            detection_results: List of detection results
            raised_tickets: List of raised tickets
            
        Returns:
            List of successfully sent notifications
        """
        sent_notifications = []
        
        # Create ticket lookup
        ticket_lookup = {
            (ticket.snode, ticket.kpi_name): ticket 
            for ticket in raised_tickets
        }
        
        for result in detection_results:
            try:
                # Get corresponding ticket
                ticket_info = ticket_lookup.get((result.snode, result.kpi_name))
                
                # Send notification
                notification_result = self.email_service.send_anomaly_notification(
                    detection_result=result,
                    ticket_info=ticket_info
                )
                
                if notification_result:
                    sent_notifications.append(notification_result)
                    self.logger.info(
                        f"Sent notification for {result.snode}/{result.kpi_name}"
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to send notification for {result.snode}: {e}")
                self.metrics.errors_count += 1
        
        return sent_notifications
    
    def run_health_check(self) -> dict:
        """
        Run system health check.
        
        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Database health
        try:
            db_status = self.database.get_connection_status()
            health_status['components']['database'] = {
                'status': 'healthy' if db_status['connected'] else 'unhealthy',
                'details': db_status
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Ticketing system health
        try:
            ticket_status = self.ticketing_service.test_connection()
            health_status['components']['ticketing'] = {
                'status': 'healthy' if ticket_status else 'unhealthy'
            }
        except Exception as e:
            health_status['components']['ticketing'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Email service health
        try:
            email_status = self.email_service.test_connection()
            health_status['components']['email'] = {
                'status': 'healthy' if email_status else 'unhealthy'
            }
        except Exception as e:
            health_status['components']['email'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Overall status
        unhealthy_components = [
            comp for comp, status in health_status['components'].items()
            if status['status'] == 'unhealthy'
        ]
        
        if unhealthy_components:
            health_status['overall_status'] = 'degraded'
            health_status['unhealthy_components'] = unhealthy_components
        
        return health_status


def main():
    """Main entry point for the monitoring script."""
    parser = argparse.ArgumentParser(
        description="Network Performance AI Anomaly Detection System"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--kpi',
        type=str,
        help='Specific KPI to monitor'
    )
    
    parser.add_argument(
        '--node',
        type=str,
        help='Specific node to monitor'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run health check only'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run detection only without raising tickets or sending emails'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = MonitoringOrchestrator(config_path=args.config)
        
        if args.health_check:
            # Run health check
            health_status = orchestrator.run_health_check()
            print("System Health Status:")
            print(f"Overall Status: {health_status['overall_status']}")
            
            for component, status in health_status['components'].items():
                print(f"{component.title()}: {status['status']}")
                if status['status'] == 'unhealthy' and 'error' in status:
                    print(f"  Error: {status['error']}")
            
            return 0 if health_status['overall_status'] == 'healthy' else 1
        
        else:
            # Run monitoring cycle
            if args.dry_run:
                orchestrator.logger.info("Running in dry-run mode - no tickets or emails will be sent")
                # TODO: Implement dry-run mode
            
            metrics = orchestrator.run_complete_cycle(
                kpi_filter=args.kpi,
                node_filter=args.node
            )
            
            print("Monitoring Cycle Results:")
            print(f"Nodes Monitored: {metrics.total_nodes_monitored}")
            print(f"KPIs Processed: {metrics.kpis_processed}")
            print(f"Anomalies Detected: {metrics.anomalies_detected}")
            print(f"Tickets Raised: {metrics.tickets_raised}")
            print(f"Emails Sent: {metrics.emails_sent}")
            print(f"Processing Time: {metrics.processing_time_seconds:.2f} seconds")
            print(f"Errors: {metrics.errors_count}")
            
            return 0 if metrics.errors_count == 0 else 1
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        return 130
    
    except Exception as e:
        print(f"Monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 