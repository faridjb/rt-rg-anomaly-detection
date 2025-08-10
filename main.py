#!/usr/bin/env python3
"""
Final Production-Ready Anomaly Detection Workflow

This script implements the complete network KPI anomaly detection workflow:
1. Loads environment variables from .env file
2. Uses email template from templates/email_templates/sg_rg_traffic_alert_compatible.html
3. Includes improved x-axis plotting for better readability
4. Email-compatible template rendering
5. Comprehensive error handling and logging

Usage:
    python3 main.py [--test] [--rg-code <code>] [--all-rg-codes] [--use-original-mode] [--node <name>] [--kpi <column>] [--max-nodes <N>]

Author: Performance Tools Team
Version: 1.0 - Final Production Release
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io
import logging
from string import Template
import shutil

# Add src to path for imports (script now resides at project root)
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file (located at project root)
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        DOTENV_AVAILABLE = True
        logger.info("✅ Environment variables loaded from .env file.")
    else:
        logger.info("📄 .env file not found. Using default environment variables or those already set.")
        DOTENV_AVAILABLE = False # Or True if you want to signify dotenv tried
except ImportError:
    print("⚠️ python-dotenv not installed. Using default environment variables.")
    logger.warning("⚠️ python-dotenv not installed. Using default environment variables or those already set.")
    DOTENV_AVAILABLE = False

# Update log level from environment if available
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(getattr(logging, log_level))

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    logger.warning("cx_Oracle not installed. Install with: pip install cx_Oracle")
    ORACLE_AVAILABLE = False


class ConfigManager:
    """Configuration manager for environment variables."""
    
    def __init__(self):
        """Load configuration from environment variables."""
        # Database Configuration
        self.db_host = os.getenv('DATABASE_HOST', '10.200.6.227')
        self.db_port = int(os.getenv('DATABASE_PORT', '1521'))
        self.db_user = os.getenv('DATABASE_USER', 'tools_ml')
        self.db_password = os.getenv('DATABASE_PASSWORD', 'Focu$2021')
        self.db_service = os.getenv('DATABASE_SERVICE', 'fcsouth.rightel.ir')
        
        # Email Configuration
        self.smtp_host = os.getenv('EMAIL_SMTP_HOST', 'mail.rightel.ir')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '465'))
        self.email_username = os.getenv('EMAIL_USERNAME', 'Performance-Dev')
        self.email_password = os.getenv('EMAIL_PASSWORD', 'Focus2021')
        self.email_from = os.getenv('EMAIL_FROM_ADDRESS', 'Performance-Dev@rightel.ir')
        
        # Recipients
        self.email_to = os.getenv('EMAIL_TO', 'Performance-Core@rightel.ir')
        self.email_cc = os.getenv('EMAIL_CC', 'Performance-Tools@rightel.ir')
        self.test_recipient = os.getenv('EMAIL_TEST_RECIPIENT', 'EX.F.Jabarimaleki@rightel.ir')
        
        # System Configuration
        self.system_id = os.getenv('SYSTEM_ID', 'CORE-AI-KPI-001')
        self.max_nodes = int(os.getenv('MAX_NODES_PROCESS', '0'))  # 0 for all nodes, or a specific number
        self.sensitivity = float(os.getenv('ANOMALY_SENSITIVITY', '0.95'))
        self.recent_hours = int(os.getenv('RECENT_HOURS_CHECK', '10')) # Updated to 10 hours
        self.min_anomalies_for_alert = int(os.getenv('MIN_ANOMALIES_FOR_ALERT', '1')) # New: min anomalies for alert
        self.historical_days = int(os.getenv('HISTORICAL_DAYS', '30'))
        self.force_test_mode = os.getenv('FORCE_TEST_MODE', 'true').lower() == 'true'  # Force alerts for testing
        
        # Low-traffic filtering (configurable)
        self.low_traffic_filter_enabled = os.getenv('LOW_TRAFFIC_FILTER_ENABLED', 'true').lower() == 'true'
        # Threshold in GB per hour for combined DL+UL
        self.low_traffic_threshold_gb_per_hour = float(os.getenv('LOW_TRAFFIC_THRESHOLD_GB_PER_HOUR', '10'))
        # Conversion: number of bytes represented by one unit in CNT1/CNT2 columns (default assumes raw bytes)
        self.traffic_bytes_per_count = float(os.getenv('TRAFFIC_BYTES_PER_COUNT', '1'))
        # Optional units helper to override bytes-per-count using common units
        low_traffic_units = os.getenv('LOW_TRAFFIC_UNITS', '').strip().lower()
        units_to_bytes = {
            'b': 1, 'bytes': 1,
            'kb': 1024, 'kbytes': 1024, 'kilobytes': 1024,
            'mb': 1024**2, 'mbytes': 1024**2, 'megabytes': 1024**2,
            'gb': 1024**3, 'gbytes': 1024**3, 'gigabytes': 1024**3,
            'tb': 1024**4, 'tbytes': 1024**4, 'terabytes': 1024**4,
        }
        if low_traffic_units in units_to_bytes:
            self.traffic_bytes_per_count = float(units_to_bytes[low_traffic_units])
        
        # Mean traffic range (hours) for mean calculations
        self.mean_traffic_range_hours = int(os.getenv('MEAN_TRAFFIC_RANGE', '72'))
        
        # Increment anomaly policy (configurable)
        # Percent threshold for caring increments (e.g., 30 means >=30%)
        self.increment_percentage = float(os.getenv('INCREAMENT_PERCENTAGE', '30'))
        # Number of days the increment must continue (steady) to be considered
        self.increment_continue_days = int(os.getenv('CONTINUE_DAYS', '3'))
        # List of RG codes for which increments are cared (comma-separated)
        inc_rg_raw = os.getenv('INCREMENT_EXCEPTION_RG', '1')
        self.increment_exception_rg = set([s.strip() for s in inc_rg_raw.split(',') if s.strip()])
        
        # Mid-traffic zero warning policy
        self.mid_traffic_threshold_gb_per_hour = float(os.getenv('MID_TRAFFIC_THRESHOLD_GB_PER_HOUR', '50'))
        self.mid_waiting_hour_for_zero_traffic = int(os.getenv('MID_WAITING_HOUR_FOR_ZERO_TRAFFIC', '3'))
        self.mid_waiting_day_for_dec_trend_traffic = int(os.getenv('MID_WAITING_DAY_FOR_DEC_TREND_TRAFFIC', '5'))
        # High-traffic zero/decline policy
        self.high_waiting_hour_for_zero_traffic = int(os.getenv('HIGH_WAITING_HOUR_FOR_ZERO_TRAFFIC', '2'))
        self.high_waiting_day_for_dec_trend_traffic = int(os.getenv('HIGH_WAITING_DAY_FOR_DEC_TREND_TRAFFIC', '5'))
        
        # Paths
        self.template_dir = Path(os.getenv('TEMPLATE_DIR', './templates'))
        self.email_template_dir = self.template_dir / "email_templates"
        self.output_dir = Path(os.getenv('CHART_OUTPUT_DIR', './output/charts'))
        self.temp_dir = Path(os.getenv('TEMP_DIR', './temp'))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.email_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default email template if it doesn't exist
        self._ensure_default_email_template()
        
    def _ensure_default_email_template(self):
        """Create default email templates if they don't exist."""
        # Only ensure the compatible template going forward
        compatible_template_path = self.email_template_dir / "sg_rg_traffic_alert_compatible.html"
        
        if not compatible_template_path.exists():
            logger.info(f"Creating compatible email template at {compatible_template_path}")
            
            # Copy from the template file if it exists in the templates directory
            template_source = self.template_dir / "sg_rg_traffic_alert_compatible.html"
            if template_source.exists():
                try:
                    shutil.copy(template_source, compatible_template_path)
                    logger.info(f"✅ Compatible email template copied successfully")
                except Exception as e:
                    logger.error(f"Failed to copy compatible email template: {e}")
            else:
                # Create a simplified compatible template
                logger.warning(f"Template source not found, creating basic compatible template")
                compatible_template = """<!DOCTYPE html PUBLIC '-//W3C//DTD XHTML 1.0 Transitional//EN' 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'>
<html xmlns='http://www.w3.org/1999/xhtml' lang='en'>
<head>
    <meta http-equiv='Content-Type' content='text/html; charset=UTF-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1.0' />
    <title>RG Interface Traffic Alert</title>
    <style type='text/css'>
        body { font-family: Arial, sans-serif; }
    </style>
</head>
<body style='margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f4f4f4;'>
    <table width='100%' border='0' cellspacing='0' cellpadding='0' style='max-width: 800px; margin: 0 auto; background-color: #ffffff; border-radius: 10px;'>
        <tr>
            <td align='center' style='padding: 20px 0; background-color: ${header_color}; border-radius: 5px 5px 0 0;'>
                <h1 style='margin: 0; color: #ffffff; font-size: 24px;'>RG Interface Traffic Alert</h1>
            </td>
        </tr>
        <tr>
            <td style='padding: 20px;'>
                <table width='100%' border='0' cellspacing='0' cellpadding='0'>
                    <tr><td style='padding-bottom: 10px;'><strong>Alert for Node:</strong> ${node_name}</td></tr>
                    <tr><td style='padding-bottom: 10px;'><strong>KPI:</strong> ${kpi_name}</td></tr>
                    <tr><td style='padding-bottom: 10px;'><strong>Time:</strong> ${alert_time}</td></tr>
                    <tr><td style='padding-bottom: 10px;'><strong>Severity:</strong> <span style='color: ${severity_color};'>${severity}</span></td></tr>
                    <tr><td style='padding-bottom: 10px;'><strong>Summary:</strong> ${summary}</td></tr>
                    <tr><td style='padding-bottom: 10px;'><strong>Details:</strong> ${additional_details}</td></tr>
                </table>
            </td>
        </tr>
        <tr>
            <td align='center' style='padding: 20px;'>
                <h2 style='font-size: 18px; color: #333333;'>Downlink Traffic</h2>
                <img src='cid:${traffic_trend}' alt='Downlink Traffic Chart' style='max-width: 100%; height: auto;'>
            </td>
        </tr>
        <tr>
            <td align='center' style='padding: 20px;'>
                <h2 style='font-size: 18px; color: #333333;'>Uplink Traffic</h2>
                <img src='cid:${performance_metrics}' alt='Uplink Traffic Chart' style='max-width: 100%; height: auto;'>
            </td>
        </tr>
        <tr>
            <td align='center' style='padding: 15px; background-color: #333333; color: #ffffff; font-size: 12px; border-radius: 0 0 5px 5px;'>
                This is an automated alert from the Network Monitoring System. System ID: ${system_id}
            </td>
        </tr>
    </table>
</body>
</html>"""
                try:
                    with open(compatible_template_path, 'w', encoding='utf-8') as f:
                        f.write(compatible_template)
                    logger.info(f"✅ Basic compatible email template created successfully")
                except Exception as e:
                    logger.error(f"Failed to create compatible email template: {e}")

    def get_db_dsn(self) -> str:
        """Get database DSN string."""
        return f"{self.db_host}:{self.db_port}/{self.db_service}"


class DatabaseService:
    """Oracle database service for KPI data retrieval."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize database service with configuration."""
        self.config = config
        self.dsn = config.get_db_dsn()
        
    def get_connection(self):
        """Get Oracle database connection."""
        if not ORACLE_AVAILABLE:
            raise ImportError("cx_Oracle not available")
        return cx_Oracle.connect(
            user=self.config.db_user, 
            password=self.config.db_password, 
            dsn=self.dsn
        )
    
    def get_distinct_rg_nodes(self, hours_back: int = 24) -> List[str]:
        """Get distinct RG node list from recent data (nodes starting with TH1VCGH1)."""
        query = """
        SELECT DISTINCT SNODE 
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
        WHERE SDATE >= SYSDATE - :hours_back/24
        AND SNODE LIKE 'TH1VCGH1%'
        ORDER BY SNODE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
    
    def get_rg_base_name(self, rg_node: str) -> str:
        """Extract RG base name without the numeric suffix.
        
        Args:
            rg_node: Full RG node name like 'TH1VCGH1_70'
            
        Returns:
            RG base name like 'TH1VCGH1'
        """
        if '_' in rg_node:
            return rg_node.split('_')[0]
        return rg_node
    
    def extract_rg_code(self, node_name: str) -> str:
        """Extract RG code from a node name (e.g., 'TH1VCGH1_70' -> '70').
        
        Args:
            node_name: Full node name like 'TH1VCGH1_70'
            
        Returns:
            RG code like '70' or None if no code is found
        """
        if '_' in node_name:
            return node_name.split('_')[-1]
        return None
    
    def get_distinct_rg_codes(self, hours_back: int = 24) -> List[str]:
        """Get distinct RG codes from recent data.
        
        Args:
            hours_back: Hours to look back for data
            
        Returns:
            List of distinct RG codes
        """
        query = """
        SELECT DISTINCT SUBSTR(SNODE, INSTR(SNODE, '_') + 1) AS RG_CODE
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
        WHERE SDATE >= SYSDATE - :hours_back/24
        AND SNODE LIKE '%\\_%' ESCAPE '\\'
        AND REGEXP_LIKE(SUBSTR(SNODE, INSTR(SNODE, '_') + 1), '^[0-9]+$')
        ORDER BY RG_CODE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
    
    def get_nodes_by_rg_code(self, rg_code: str, hours_back: int = 24) -> List[str]:
        """Get all nodes associated with a specific RG code.
        
        Args:
            rg_code: RG code like '70'
            hours_back: Hours to look back for data
            
        Returns:
            List of node names with the specified RG code
        """
        query = """
        SELECT DISTINCT SNODE
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
        WHERE SDATE >= SYSDATE - :hours_back/24
        AND SNODE LIKE '%\\_' || :rg_code ESCAPE '\\'
        ORDER BY SNODE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back, 'rg_code': rg_code})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
    
    def get_associated_cg_nodes(self, rg_base: str, hours_back: int = 24) -> List[str]:
        """Get all CG nodes associated with an RG base.
        
        For now, this returns all active CG nodes. In a more sophisticated setup,
        this could be based on actual routing relationships.
        
        Args:
            rg_base: RG base name like 'TH1VCGH1'
            hours_back: Hours to look back for active nodes
            
        Returns:
            List of associated CG node names
        """
        query = """
        SELECT DISTINCT SNODE 
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
        WHERE SDATE >= SYSDATE - :hours_back/24
        AND SNODE LIKE 'TH1CGH1%'
        ORDER BY SNODE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
    
    def get_aggregated_rg_data(self, rg_nodes: List[str], days_back: int = 30) -> pd.DataFrame:
        """Get aggregated historical KPI data for a list of RG nodes.
        
        Args:
            rg_nodes: List of RG node names to aggregate
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with aggregated traffic data
        """
        if not rg_nodes:
            return pd.DataFrame(columns=['timestamp', 'rg_downlink', 'rg_uplink', 'total_rg'])
        
        # Create IN clause for SQL query
        node_placeholders = ','.join([f':node_{i}' for i in range(len(rg_nodes))])
        
        query = f"""
        SELECT SDATE,
               SUM(CNT1_167774004) AS rg_downlink,
               SUM(CNT2_167774004) AS rg_uplink,
               SUM(CNT1_167774004 + CNT2_167774004) AS total_rg
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SNODE IN ({node_placeholders})
        AND SDATE >= SYSDATE - :days_back
        GROUP BY SDATE
        ORDER BY SDATE
        """
        
        # Prepare parameters
        params = {'days_back': days_back}
        for i, node in enumerate(rg_nodes):
            params[f'node_{i}'] = node
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Fetch results
            columns = ['timestamp', 'rg_downlink', 'rg_uplink', 'total_rg']
            results = cursor.fetchall()
            
            if not results:
                return pd.DataFrame(columns=columns)
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Fill missing values with interpolation
            for col in ['rg_downlink', 'rg_uplink', 'total_rg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate()
                
            return df
            
    def get_aggregated_data_by_rg_code(self, rg_code: str, days_back: int = 30) -> pd.DataFrame:
        """Get aggregated historical KPI data for all nodes with a specific RG code.
        
        Args:
            rg_code: RG code like '70'
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with aggregated traffic data for this RG code
        """
        query = """
        SELECT SDATE,
               SUM(CNT1_167774004) AS rg_downlink,
               SUM(CNT2_167774004) AS rg_uplink,
               SUM(CNT1_167774004 + CNT2_167774004) AS total_rg
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SNODE LIKE '%\\_' || :rg_code ESCAPE '\\'
        AND SDATE >= SYSDATE - :days_back
        GROUP BY SDATE
        ORDER BY SDATE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'rg_code': rg_code, 'days_back': days_back})
            
            # Fetch results
            columns = ['timestamp', 'rg_downlink', 'rg_uplink', 'total_rg']
            results = cursor.fetchall()
            
            if not results:
                return pd.DataFrame(columns=columns)
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Fill missing values with interpolation
            for col in ['rg_downlink', 'rg_uplink', 'total_rg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate()
                
            return df

    def get_distinct_nodes(self, hours_back: int = 24) -> List[str]:
        """Get distinct node list from recent data."""
        query = """
        SELECT DISTINCT SNODE 
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
        WHERE SDATE >= SYSDATE - :hours_back/24
        ORDER BY SNODE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
    
    def get_node_historical_data(self, node_name: str, days_back: int = 30) -> pd.DataFrame:
        """Get historical KPI data for a specific node."""
        query = """
        SELECT SDATE,
               CNT1_167774004 AS rg_downlink,
               CNT2_167774004 AS rg_uplink,
               (CNT1_167774004 + CNT2_167774004) AS total_rg
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SNODE = :node_name 
        AND SDATE >= SYSDATE - :days_back
        ORDER BY SDATE
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'node_name': node_name, 'days_back': days_back})
            
            # Fetch results
            columns = ['timestamp', 'rg_downlink', 'rg_uplink', 'total_rg']
            results = cursor.fetchall()
            
            if not results:
                return pd.DataFrame(columns=columns)
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Fill missing values with interpolation
            for col in ['rg_downlink', 'rg_uplink', 'total_rg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate()
                
            return df
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.fetchone()
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_rg_to_cg_traffic(self, rg_code: str, cg_nodes: List[str], days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """Get traffic data from RG nodes with specific code to each CG node.
        
        Args:
            rg_code: The RG code to filter nodes by
            cg_nodes: List of CG node names
            days_back: Number of days of historical data
            
        Returns:
            Dictionary with two DataFrames: 'dl' for downlink and 'ul' for uplink traffic
        """
        if not cg_nodes:
            return {'dl': pd.DataFrame(), 'ul': pd.DataFrame()}
        
        # Get all RG nodes with this code
        rg_nodes = self.get_nodes_by_rg_code(rg_code, hours_back=days_back * 24)
        
        if not rg_nodes:
            return {'dl': pd.DataFrame(), 'ul': pd.DataFrame()}
            
        # Create IN clause for CG nodes
        cg_placeholders = ','.join([f':cg_{i}' for i in range(len(cg_nodes))])
        
        # Use a simpler query that doesn't rely on RG nodes
        # Just filter by CG nodes and get the traffic data
        query = f"""
        SELECT SDATE as timestamp,
               SNODE as cg_node,
               SUM(CNT1_167774004) AS rg_downlink,
               SUM(CNT2_167774004) AS rg_uplink
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SNODE IN ({cg_placeholders})
        AND SDATE >= SYSDATE - :days_back
        GROUP BY SDATE, SNODE
        ORDER BY SDATE, SNODE
        """
        
        # Prepare parameters
        params = {'days_back': days_back}
        for i, node in enumerate(cg_nodes):
            params[f'cg_{i}'] = node
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                # Fetch results
                columns = ['timestamp', 'cg_node', 'rg_downlink', 'rg_uplink']
                results = cursor.fetchall()
                
                if not results:
                    return {'dl': pd.DataFrame(), 'ul': pd.DataFrame()}
                
                # Create DataFrame
                df = pd.DataFrame(results, columns=columns)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert numeric columns
                for col in ['rg_downlink', 'rg_uplink']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create separate DFs for DL and UL
                df_dl = df[['timestamp', 'cg_node', 'rg_downlink']].copy()
                df_ul = df[['timestamp', 'cg_node', 'rg_uplink']].copy()
                
                return {'dl': df_dl, 'ul': df_ul}
                
        except Exception as e:
            logger.error(f"Error getting RG to CG traffic: {e}", exc_info=True)
            return {'dl': pd.DataFrame(), 'ul': pd.DataFrame()}

    def get_cg_nodes_for_rg_code(self, rg_code: str, hours_back: int = 24) -> List[str]:
        """Get distinct CG nodes that belong to a specific RG code.
        
        Matches nodes that have suffix _<rg_code> and contain 'CG' in the name.
        """
        query = """
        SELECT DISTINCT SNODE
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SDATE >= SYSDATE - :hours_back/24
          AND SNODE LIKE '%\\_' || :rg_code ESCAPE '\\'
          AND REGEXP_LIKE(SNODE, 'CG', 'i')
        ORDER BY SNODE
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, {'hours_back': hours_back, 'rg_code': rg_code})
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]] 

    def get_aggregated_cg_by_rg_code(self, rg_code: str, days_back: int = 30) -> pd.DataFrame:
        """Aggregate historical KPI data for all CG nodes associated with the RG code.
        
        This restricts aggregation to CG nodes whose names contain 'CG' and end with _<rg_code>.
        """
        cg_nodes = self.get_cg_nodes_for_rg_code(rg_code, hours_back=days_back * 24)
        return self.get_aggregated_rg_data(cg_nodes, days_back) if cg_nodes else pd.DataFrame(columns=['timestamp','rg_downlink','rg_uplink','total_rg'])


class AnomalyDetector:
    """Prophet-based anomaly detection for network KPI data."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize anomaly detector."""
        self.sensitivity = config.sensitivity
        self.model = None
        self.forecast = None
        
    def detect_anomalies(self, df: pd.DataFrame, metric_column: str = 'total_rg') -> Dict[str, Any]:
        """Detect anomalies in time series data using Prophet."""
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available for anomaly detection")
            return {'success': False, 'error': 'Prophet not installed'}
        
        if df.empty or len(df) < 10:
            return {'success': False, 'error': 'Insufficient data for analysis'}
        
        try:
            # Prepare data for Prophet
            prophet_df = df[['timestamp', metric_column]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 10:
                return {'success': False, 'error': 'Insufficient valid data points'}
            
            # Configure Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=self.sensitivity,
                changepoint_prior_scale=0.05,
                uncertainty_samples=100
            )
            
            # Fit model
            model.fit(prophet_df)
            
            # Create forecast
            forecast = model.predict(prophet_df[['ds']])
            
            # Identify anomalies
            forecast['anomaly'] = (prophet_df['y'] < forecast['yhat_lower']) | (prophet_df['y'] > forecast['yhat_upper'])
            
            # Merge anomalies back into the original dataframe
            merged_df = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'anomaly']], left_on='timestamp', right_on='ds', how='left')
            merged_df = merged_df.drop(columns=['ds'])

            self.model = model
            self.forecast = forecast

            return {
                'success': True,
                'data_with_anomalies': merged_df,
                'forecast_df': forecast,
                'model': model
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def detect_anomalies_dl_ul(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in both downlink and uplink traffic.
        
        Args:
            df: DataFrame with timestamp, rg_downlink, and rg_uplink columns
            
        Returns:
            Dictionary with detection results for both metrics
        """
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available for anomaly detection")
            return {'success': False, 'error': 'Prophet not installed'}
        
        if df.empty or len(df) < 10:
            return {'success': False, 'error': 'Insufficient data for analysis'}
        
        try:
            # Detect anomalies in downlink traffic
            dl_result = self.detect_anomalies(df, metric_column='rg_downlink')
            
            # Detect anomalies in uplink traffic
            ul_result = self.detect_anomalies(df, metric_column='rg_uplink')
            
            if not dl_result['success'] or not ul_result['success']:
                error_msg = []
                if not dl_result['success']:
                    error_msg.append(f"DL analysis: {dl_result['error']}")
                if not ul_result['success']:
                    error_msg.append(f"UL analysis: {ul_result['error']}")
                return {'success': False, 'error': ', '.join(error_msg)}
            
            # Merge the anomaly results into a single dataframe
            df_with_anomalies = df.copy()
            
            # Add DL anomaly flags
            df_dl_anomalies = dl_result['data_with_anomalies']
            df_with_anomalies['dl_anomaly'] = False
            df_with_anomalies['dl_yhat'] = np.nan
            df_with_anomalies['dl_yhat_lower'] = np.nan
            df_with_anomalies['dl_yhat_upper'] = np.nan
            
            for idx, row in df_dl_anomalies.iterrows():
                mask = df_with_anomalies['timestamp'] == row['timestamp']
                if any(mask):
                    df_with_anomalies.loc[mask, 'dl_anomaly'] = row['anomaly']
                    df_with_anomalies.loc[mask, 'dl_yhat'] = row['yhat']
                    df_with_anomalies.loc[mask, 'dl_yhat_lower'] = row['yhat_lower']
                    df_with_anomalies.loc[mask, 'dl_yhat_upper'] = row['yhat_upper']
            
            # Add UL anomaly flags
            df_ul_anomalies = ul_result['data_with_anomalies']
            df_with_anomalies['ul_anomaly'] = False
            df_with_anomalies['ul_yhat'] = np.nan
            df_with_anomalies['ul_yhat_lower'] = np.nan
            df_with_anomalies['ul_yhat_upper'] = np.nan
            
            for idx, row in df_ul_anomalies.iterrows():
                mask = df_with_anomalies['timestamp'] == row['timestamp']
                if any(mask):
                    df_with_anomalies.loc[mask, 'ul_anomaly'] = row['anomaly']
                    df_with_anomalies.loc[mask, 'ul_yhat'] = row['yhat']
                    df_with_anomalies.loc[mask, 'ul_yhat_lower'] = row['yhat_lower']
                    df_with_anomalies.loc[mask, 'ul_yhat_upper'] = row['yhat_upper']
            
            # Add combined anomaly flag
            df_with_anomalies['anomaly'] = df_with_anomalies['dl_anomaly'] | df_with_anomalies['ul_anomaly']
            
            return {
                'success': True,
                'data_with_anomalies': df_with_anomalies,
                'dl_forecast_df': dl_result['forecast_df'],
                'ul_forecast_df': ul_result['forecast_df'],
                'dl_model': dl_result['model'],
                'ul_model': ul_result['model']
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


class VisualizationService:
    """Service for creating and saving visualizations."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize visualization service."""
        self.output_dir = config.output_dir
        
        # Try to set a nice style, fall back to default if not available
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    # Use default matplotlib style if seaborn is not available
                    plt.style.use('default')
                    logger.warning("Seaborn styles not available, using default matplotlib style")

    def _format_plot(self, ax, title: str, df_len: int) -> None:
        """Apply common formatting to a plot."""
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=10)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Dynamic x-axis ticks based on data timespan
        if df_len > 0: # Ensure df_len is positive
            first_date = plt.gca().lines[0].get_xdata()[0] # Get first date from plot data
            last_date = plt.gca().lines[0].get_xdata()[-1] # Get last date from plot data
            
            # Ensure first_date and last_date are valid Matplotlib date numbers
            if not (np.isreal(first_date) and np.isreal(last_date)):
                 logger.warning("Invalid date data for plot formatting. Skipping dynamic ticks.")
                 return

            # Convert matplotlib dates to datetime objects if necessary
            # Matplotlib dates are days since 0001-01-01 UTC + 1
            # So we add one day to the ordinal to get the correct date
            if isinstance(first_date, (int, float)) and isinstance(last_date, (int, float)):
                 try:
                    first_dt = mdates.num2date(first_date)
                    last_dt = mdates.num2date(last_date)
                 except ValueError as e:
                    logger.warning(f"Could not convert numerical dates to datetime: {e}. Skipping dynamic ticks.")
                    return
            else: # Assuming they are already datetime-like or can be converted
                try:
                    first_dt = pd.to_datetime(first_date)
                    last_dt = pd.to_datetime(last_date)
                except Exception as e:
                    logger.warning(f"Could not process date data for plot formatting: {e}. Skipping dynamic ticks.")
                    return

            time_span_days = (last_dt - first_dt).total_seconds() / (24 * 3600)

            from matplotlib.ticker import MaxNLocator # Correct import

            if time_span_days > 7:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_span_days / 10)))) # At most 10 ticks
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            elif time_span_days > 1:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12)) # Every 12 hours for <= 7 days
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            else: # <= 1 day
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(24 / 10)))) # At most 10 ticks
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1)) # Minor ticks every hour
            # Ensure MaxNLocator is used to limit total ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))


    def plot_kpi_with_anomalies(
        self, 
        df: pd.DataFrame, 
        node_name: str, 
        metric_column: str = 'total_rg',
        forecast_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, str]]:
        """Plot KPI data with anomalies and forecast for two time ranges using Seaborn."""
        if df.empty:
            logger.warning(f"No data to plot for node {node_name}, metric {metric_column}.")
            return None

        # Create two plots: one for 36 hours and one for 30 days
        plot_paths = []
        time_ranges = [('36 Hours', 1.5), ('30 Days', 30)]
        
        for label, days in time_ranges:
            plt.figure(figsize=(15, 7))
            ax = plt.gca()

            # Filter data for the specific time range
            end_time = df['timestamp'].max()
            start_time = end_time - timedelta(days=days)
            df_range = df[df['timestamp'] >= start_time]
            forecast_range = forecast_df[forecast_df['ds'] >= start_time] if forecast_df is not None else None

            if df_range.empty:
                logger.warning(f"No data in {label} range for node {node_name}, metric {metric_column}.")
                plt.close()
                continue

            # Plot actual data
            sns.lineplot(x='timestamp', y=metric_column, data=df_range, label='Actual KPI', ax=ax, color='blue', linewidth=1.5)

            # Plot mean
            mean_value = df_range[metric_column].mean()
            ax.axhline(y=mean_value, color='green', linestyle='-', linewidth=1.5, label='Mean', alpha=0.7)

            # Plot overall trend (rolling mean for 12 periods)
            df_range['trend'] = df_range[metric_column].rolling(window=12, center=True).mean()
            sns.lineplot(x='timestamp', y='trend', data=df_range, label='12-Period Trend', ax=ax, color='purple', linewidth=1.5, linestyle='--')

            # Plot forecast if available
            if forecast_range is not None and not forecast_range.empty:
                sns.lineplot(x='ds', y='yhat', data=forecast_range, label='Forecast (yhat)', ax=ax, color='orange', linestyle='--', linewidth=1.5)
                ax.fill_between(forecast_range['ds'], forecast_range['yhat_lower'], forecast_range['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')

            # Highlight anomalies
            anomalies = df_range[df_range['anomaly'] == True]
            if not anomalies.empty:
                sns.scatterplot(x='timestamp', y=metric_column, data=anomalies, color='red', s=100, label='Anomaly', ax=ax, marker='o', zorder=5)
            
            df_len = len(df_range['timestamp']) if 'timestamp' in df_range else 0
            self._format_plot(ax, f"KPI Anomaly Detection for {node_name} - {metric_column} ({label})", df_len)
            
            # Save plot
            plot_filename = f"{node_name}_{metric_column}_anomaly_plot_{label.replace(' ', '_')}.png"
            plot_path = self.output_dir / plot_filename
            try:
                plt.savefig(plot_path, dpi=100, bbox_inches='tight') # Lower DPI for email size
                plt.close()
                logger.info(f"Plot saved to {plot_path}")
                plot_paths.append(str(plot_path))
            except Exception as e:
                logger.error(f"Failed to save plot {plot_path}: {e}")
                plt.close()

        # Return a dictionary of plot paths
        plot_files = {'30_Days': None, '36_Hours': None}
        for p_path in plot_paths:
            if '30_Days' in Path(p_path).name:
                plot_files['30_Days'] = p_path
            elif '36_Hours' in Path(p_path).name:
                plot_files['36_Hours'] = p_path
        return plot_files
        
    def plot_rg_code_analysis(
        self,
        rg_code: str,
        df_total: pd.DataFrame,
        df_total_with_anomalies: pd.DataFrame,
        df_rg_dl: pd.DataFrame = None,
        df_rg_ul: pd.DataFrame = None,
        time_range: str = '30 Days',
        df_per_cg_dl: Optional[pd.DataFrame] = None,
        df_per_cg_ul: Optional[pd.DataFrame] = None,
        mid_trend_days: Optional[int] = None,
        high_trend_days: Optional[int] = None,
    ) -> Optional[Dict[str, str]]:
        """Plot RG code traffic analysis with separate plots for DL and UL traffic.
        
        Creates 4 plots:
        (1) Downlink traffic of RG code to each CG separately (multiple lines with legend)
        (2) Uplink traffic of RG code to each CG separately (multiple lines with legend)
        (3) Aggregated DL traffic for this RG code with anomaly detection
        (4) Aggregated UL traffic for this RG code with anomaly detection
        
        If per-CG DataFrames are not provided, falls back to total DL/UL in (1) and (2).
        
        Returns:
            Dictionary with paths to the generated plot files
        """
        if df_total.empty:
            logger.warning(f"No data to plot for RG code {rg_code}.")
            return None
            
        days = 1.5 if time_range == '36 Hours' else 30
        
        end_time = df_total['timestamp'].max()
        start_time = end_time - timedelta(days=days)
        
        df_range = df_total[df_total['timestamp'] >= start_time]
        df_anomalies_range = df_total_with_anomalies[df_total_with_anomalies['timestamp'] >= start_time]
        
        if df_range.empty:
            logger.warning(f"No data in {time_range} range for RG code {rg_code}.")
            return None
            
        plot_paths = {}
        
        # Plot 1: DL per-CG lines
        plt.figure(figsize=(15, 8))
        ax1 = plt.gca()
        if df_per_cg_dl is not None and not df_per_cg_dl.empty and 'cg_node' in df_per_cg_dl.columns:
            df_cg_dl_range = df_per_cg_dl[df_per_cg_dl['timestamp'] >= start_time]
            for cg_name, df_cg in df_cg_dl_range.groupby('cg_node'):
                sns.lineplot(x='timestamp', y='rg_downlink', data=df_cg, label=f'DL {cg_name}', ax=ax1, linewidth=1.5)
            ax1.set_title(f"RG {rg_code} Downlink to CGs ({time_range})", fontsize=16)
        else:
            sns.lineplot(x='timestamp', y='rg_downlink', data=df_range, label=f'Total DL Traffic', ax=ax1, color='blue', linewidth=1.5)
            ax1.set_title(f"Total Downlink Traffic for RG {rg_code} ({time_range})", fontsize=16)
        ax1.set_ylabel("Downlink Traffic", fontsize=12)
        ax1.set_xlabel("Time", fontsize=12)
        ax1.legend(fontsize=10)
        self._format_plot(ax1, f"RG {rg_code} Downlink to CGs ({time_range})", len(df_range))
        
        dl_cg_plot_filename = f"RG_{rg_code}_DL_per_CG_{time_range.replace(' ', '_')}.png"
        dl_cg_plot_path = self.output_dir / dl_cg_plot_filename
        plt.savefig(dl_cg_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths['dl_cg'] = str(dl_cg_plot_path)
        
        # Plot 2: UL per-CG lines
        plt.figure(figsize=(15, 8))
        ax2 = plt.gca()
        if df_per_cg_ul is not None and not df_per_cg_ul.empty and 'cg_node' in df_per_cg_ul.columns:
            df_cg_ul_range = df_per_cg_ul[df_per_cg_ul['timestamp'] >= start_time]
            for cg_name, df_cg in df_cg_ul_range.groupby('cg_node'):
                sns.lineplot(x='timestamp', y='rg_uplink', data=df_cg, label=f'UL {cg_name}', ax=ax2, linewidth=1.5)
            ax2.set_title(f"RG {rg_code} Uplink to CGs ({time_range})", fontsize=16)
        else:
            sns.lineplot(x='timestamp', y='rg_uplink', data=df_range, label=f'Total UL Traffic', ax=ax2, color='green', linewidth=1.5)
            ax2.set_title(f"Total Uplink Traffic for RG {rg_code} ({time_range})", fontsize=16)
        ax2.set_ylabel("Uplink Traffic", fontsize=12)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.legend(fontsize=10)
        self._format_plot(ax2, f"RG {rg_code} Uplink to CGs ({time_range})", len(df_range))
        
        ul_cg_plot_filename = f"RG_{rg_code}_UL_per_CG_{time_range.replace(' ', '_')}.png"
        ul_cg_plot_path = self.output_dir / ul_cg_plot_filename
        plt.savefig(ul_cg_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths['ul_cg'] = str(ul_cg_plot_path)
        
        # Plot 3: Aggregated DL traffic with anomaly detection
        plt.figure(figsize=(15, 8))
        ax3 = plt.gca()

        # Decide which dataframe to plot for DL
        dl_line_df = None
        if df_rg_dl is not None and not df_rg_dl.empty:
            dl_line_df = df_rg_dl[df_rg_dl['timestamp'] >= start_time]
        else:
            dl_line_df = df_total[df_total['timestamp'] >= start_time]

        if dl_line_df is not None and not dl_line_df.empty:
            sns.lineplot(x='timestamp', y='rg_downlink', data=dl_line_df, label=f'RG {rg_code} DL Traffic', ax=ax3, color='blue', linewidth=1.5)
            # Add daily mean trendline (DL)
            try:
                dl_daily_mean = dl_line_df.set_index('timestamp')['rg_downlink'].resample('D').mean().rename('daily_mean').reset_index()
                if not dl_daily_mean.empty:
                    # Ignore first/last points for smoother trendline visibility
                    dl_plot_df = dl_daily_mean.iloc[1:-1] if len(dl_daily_mean) > 2 else dl_daily_mean
                    sns.lineplot(x='timestamp', y='daily_mean', data=dl_plot_df, label='DL Daily Mean', ax=ax3, color='orange', linestyle='--', linewidth=3)
                # Overlay recent decrement trendline(s) in red solid
                if mid_trend_days and mid_trend_days > 0 and not dl_daily_mean.empty:
                    dl_mid_seg = dl_daily_mean.tail(mid_trend_days)
                    if len(dl_mid_seg) >= 2:
                        sns.lineplot(x='timestamp', y='daily_mean', data=dl_mid_seg, label=f'Mid-tier Recent Trend ({mid_trend_days}d)', ax=ax3, color='red', linestyle='-', linewidth=2)
                if high_trend_days and high_trend_days > 0 and not dl_daily_mean.empty:
                    dl_high_seg = dl_daily_mean.tail(high_trend_days)
                    if len(dl_high_seg) >= 2:
                        sns.lineplot(x='timestamp', y='daily_mean', data=dl_high_seg, label=f'High-tier Recent Trend ({high_trend_days}d)', ax=ax3, color='red', linestyle='-', linewidth=2)
            except Exception as e:
                logger.warning(f"Failed to compute DL daily mean trendline: {e}")

        # Add DL anomalies if available (align to the same dataframe as the line)
        if 'dl_anomaly' in df_anomalies_range.columns and dl_line_df is not None and not dl_line_df.empty:
            dl_anom_merge = pd.merge(dl_line_df[['timestamp', 'rg_downlink']],
                                     df_anomalies_range[['timestamp', 'dl_anomaly']],
                                     on='timestamp', how='left')
            dl_anomalies = dl_anom_merge[dl_anom_merge['dl_anomaly'] == True]
            if not dl_anomalies.empty:
                sns.scatterplot(x='timestamp', y='rg_downlink', data=dl_anomalies, color='red', s=100, label='Anomaly', ax=ax3, marker='o')

        ax3.set_title(f"Aggregated Downlink Traffic from RG {rg_code} with Anomaly Detection ({time_range})", fontsize=16)
        ax3.set_ylabel("Downlink Traffic", fontsize=12)
        ax3.set_xlabel("Time", fontsize=12)
        ax3.legend(fontsize=10)
        self._format_plot(ax3, f"Aggregated Downlink Traffic from RG {rg_code} ({time_range})", len(dl_line_df) if dl_line_df is not None else 0)

        dl_agg_plot_filename = f"RG_{rg_code}_DL_aggregated_{time_range.replace(' ', '_')}.png"
        dl_agg_plot_path = self.output_dir / dl_agg_plot_filename
        plt.savefig(dl_agg_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths['dl_agg'] = str(dl_agg_plot_path)
        
        # Plot 4: Aggregated UL traffic with anomaly detection
        plt.figure(figsize=(15, 8))
        ax4 = plt.gca()
        
        # Decide which dataframe to plot for UL
        ul_line_df = None
        if df_rg_ul is not None and not df_rg_ul.empty:
            ul_line_df = df_rg_ul[df_rg_ul['timestamp'] >= start_time]
        else:
            ul_line_df = df_total[df_total['timestamp'] >= start_time]

        if ul_line_df is not None and not ul_line_df.empty:
            sns.lineplot(x='timestamp', y='rg_uplink', data=ul_line_df, label=f'RG {rg_code} UL Traffic', ax=ax4, color='green', linewidth=1.5)
            # Add daily mean trendline (UL)
            try:
                ul_daily_mean = ul_line_df.set_index('timestamp')['rg_uplink'].resample('D').mean().rename('daily_mean').reset_index()
                if not ul_daily_mean.empty:
                    # Ignore first/last points for smoother trendline visibility
                    ul_plot_df = ul_daily_mean.iloc[1:-1] if len(ul_daily_mean) > 2 else ul_daily_mean
                    sns.lineplot(x='timestamp', y='daily_mean', data=ul_plot_df, label='UL Daily Mean', ax=ax4, color='brown', linestyle='--', linewidth=3)
                # Overlay recent decrement trendline(s) in red solid
                if mid_trend_days and mid_trend_days > 0 and not ul_daily_mean.empty:
                    ul_mid_seg = ul_daily_mean.tail(mid_trend_days)
                    if len(ul_mid_seg) >= 2:
                        sns.lineplot(x='timestamp', y='daily_mean', data=ul_mid_seg, label=f'Mid-tier Recent Trend ({mid_trend_days}d)', ax=ax4, color='red', linestyle='-', linewidth=2)
                if high_trend_days and high_trend_days > 0 and not ul_daily_mean.empty:
                    ul_high_seg = ul_daily_mean.tail(high_trend_days)
                    if len(ul_high_seg) >= 2:
                        sns.lineplot(x='timestamp', y='daily_mean', data=ul_high_seg, label=f'High-tier Recent Trend ({high_trend_days}d)', ax=ax4, color='red', linestyle='-', linewidth=2)
            except Exception as e:
                logger.warning(f"Failed to compute UL daily mean trendline: {e}")

        # Add UL anomalies if available (align to the same dataframe as the line)
        if 'ul_anomaly' in df_anomalies_range.columns and ul_line_df is not None and not ul_line_df.empty:
            ul_anom_merge = pd.merge(ul_line_df[['timestamp', 'rg_uplink']],
                                     df_anomalies_range[['timestamp', 'ul_anomaly']],
                                     on='timestamp', how='left')
            ul_anomalies = ul_anom_merge[ul_anom_merge['ul_anomaly'] == True]
            if not ul_anomalies.empty:
                sns.scatterplot(x='timestamp', y='rg_uplink', data=ul_anomalies, color='red', s=100, label='Anomaly', ax=ax4, marker='o')

        ax4.set_title(f"Aggregated Uplink Traffic from RG {rg_code} with Anomaly Detection ({time_range})", fontsize=16)
        ax4.set_ylabel("Uplink Traffic", fontsize=12)
        ax4.set_xlabel("Time", fontsize=12)
        ax4.legend(fontsize=10)
        self._format_plot(ax4, f"Aggregated Uplink Traffic from RG {rg_code} ({time_range})", len(ul_line_df) if ul_line_df is not None else 0)

        ul_agg_plot_filename = f"RG_{rg_code}_UL_aggregated_{time_range.replace(' ', '_')}.png"
        ul_agg_plot_path = self.output_dir / ul_agg_plot_filename
        plt.savefig(ul_agg_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths['ul_agg'] = str(ul_agg_plot_path)
        
        logger.info(f"RG code analysis plots saved to {self.output_dir}")
        return plot_paths

    def plot_prophet_components(self, model, forecast, node_name: str, metric_column: str) -> Optional[str]:
        """Plot Prophet model components."""
        if not PROPHET_AVAILABLE:
            return None
        try:
            fig = model.plot_components(forecast)
            plt.suptitle(f"Prophet Model Components for {node_name} - {metric_column}", fontsize=16, y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle
            
            components_filename = f"{node_name}_{metric_column}_prophet_components.png"
            components_path = self.output_dir / components_filename
            fig.savefig(components_path, dpi=100) # Lower DPI for email size
            plt.close(fig)
            logger.info(f"Prophet components plot saved to {components_path}")
            return str(components_path)
        except Exception as e:
            logger.error(f"Failed to plot Prophet components for {node_name}: {e}")
            plt.close()
            return None


class EmailService:
    """Service for composing and sending email alerts."""
    
    def __init__(self, config: ConfigManager):
        """Initialize email service."""
        self.config = config
        self.email_template_path = config.email_template_dir / "sg_rg_traffic_alert_compatible.html"
        # If compatible template is not available, built-in minimal template will be used at send time

    def _load_email_template(self) -> Optional[Template]:
        """Load the HTML email template."""
        try:
            with open(self.email_template_path, 'r', encoding='utf-8') as f:
                return Template(f.read())
        except FileNotFoundError:
            logger.error(f"Email template not found at {self.email_template_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading email template: {e}")
            return None

    def _create_email_compatible_body(self, node_name: str, kpi_name: str, alert_time: str,
                                     severity: str, summary: str, plot_cid: str,
                                     additional_details: str = "N/A") -> str:
        """Creates an email body string using basic HTML tables for compatibility."""
        
        template_str = """
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Network KPI Anomaly Alert</title>
        </head>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f4f4f4;">
                <tr>
                    <td align="center" style="padding: 20px 0;">
                        <table width="600" border="0" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border: 1px solid #dddddd; border-radius: 5px;">
                            <!-- Header -->
                            <tr>
                                <td align="center" style="padding: 20px 0; background-color: ${header_color}; border-bottom: 1px solid #cccccc; border-radius: 5px 5px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 24px;">Network KPI Anomaly Alert</h1>
                                </td>
                            </tr>
                            <!-- Body -->
                            <tr>
                                <td style="padding: 20px;">
                                    <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                        <tr><td style="padding-bottom: 10px;"><strong>Alert for Node:</strong> ${node_name}</td></tr>
                                        <tr><td style="padding-bottom: 10px;"><strong>KPI:</strong> ${kpi_name}</td></tr>
                                        <tr><td style="padding-bottom: 10px;"><strong>Time:</strong> ${alert_time}</td></tr>
                                        <tr><td style="padding-bottom: 10px;"><strong>Severity:</strong> <span style="color: ${severity_color}; font-weight: bold;">${severity}</span></td></tr>
                                        <tr><td style="padding-bottom: 10px;"><strong>Summary:</strong> ${summary}</td></tr>
                                        <tr><td style="padding-bottom: 10px;"><strong>Details:</strong> ${additional_details}</td></tr>
                                    </table>
                                </td>
                            </tr>
                            <!-- Chart -->
                            <tr>
                                <td align="center" style="padding: 0 20px 20px 20px;">
                                    <h2 style="font-size: 18px; color: #333333; margin-top: 0;">Anomaly Chart</h2>
                                    <img src="cid:${plot_cid}" alt="Anomaly Chart" style="max-width: 100%; height: auto; border: 1px solid #cccccc;">
                                </td>
                            </tr>
                            <!-- Footer -->
                            <tr>
                                <td align="center" style="padding: 15px; background-color: #333333; color: #ffffff; font-size: 12px; border-radius: 0 0 5px 5px;">
                                    This is an automated alert from the Network Monitoring System. System ID: ${system_id}
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        severity_color_map = {
            "CRITICAL": "#D32F2F", # Red
            "WARNING": "#FFA000",  # Orange
            "INFO": "#1976D2"     # Blue
        }
        header_color_map = {
            "CRITICAL": "#B71C1C", # Darker Red
            "WARNING": "#F57C00",  # Darker Orange
            "INFO": "#0D47A1"     # Darker Blue
        }

        email_template = Template(template_str)
        
        html_body = email_template.substitute(
            node_name=node_name,
            kpi_name=kpi_name,
            alert_time=alert_time,
            severity=severity.upper(),
            severity_color=severity_color_map.get(severity.upper(), "#333333"), # Default to dark grey
            header_color=header_color_map.get(severity.upper(), "#555555"),     # Default to darker grey
            summary=summary,
            plot_cid=plot_cid,
            additional_details=additional_details,
            system_id=self.config.system_id
        )
        return html_body

    def send_alert_email(self, subject: str, body_params: Dict[str, str], plot_path: Optional[str], plot_path2: Optional[str] = None, to_override: Optional[List[str]] = None, cc_override: Optional[List[str]] = None) -> bool:
        """Send an email alert with embedded plots (up to two images, for compatible template)."""
        
        html_template = self._load_email_template()
        if html_template is None:
            logger.error("Cannot send email: HTML template failed to load. Falling back to built-in template.")
            if 'plot_cid' not in body_params and plot_path:
                body_params['plot_cid'] = Path(plot_path).name if plot_path else "plot_image"

            html_body = self._create_email_compatible_body(
                node_name=body_params.get('node_name', 'N/A'),
                kpi_name=body_params.get('kpi_name', 'N/A'),
                alert_time=body_params.get('alert_time', 'N/A'),
                severity=body_params.get('severity', 'INFO'),
                summary=body_params.get('summary', 'No summary provided.'),
                plot_cid=body_params.get('plot_cid', "plot_image"),
                additional_details=body_params.get('additional_details', 'N/A')
            )
        else:
            if 'plot_cid' not in body_params and plot_path:
                body_params['plot_cid'] = Path(plot_path).name if plot_path else "plot_image"
            severity_color_map = { "CRITICAL": "#D32F2F", "WARNING": "#FFA000", "INFO": "#1976D2" }
            header_color_map = { "CRITICAL": "#B71C1C", "WARNING": "#F57C00", "INFO": "#0D47A1" }
            current_severity = body_params.get('severity', 'INFO').upper()
            body_params['severity_color'] = severity_color_map.get(current_severity, severity_color_map['INFO'])
            body_params['header_color'] = header_color_map.get(current_severity, header_color_map['INFO'])

            default_params = {
                'node_name': 'N/A', 'kpi_name': 'N/A', 'alert_time': 'N/A',
                'severity': 'INFO', 'summary': 'No summary.', 'additional_details': 'N/A',
                'system_id': self.config.system_id,
                'plot_cid': body_params.get('plot_cid', 'plot_image'),
                'rg_cg_list': body_params.get('rg_cg_list', 'N/A')
            }
            final_params = {**default_params, **body_params}
            html_body = html_template.substitute(final_params)

        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = self.config.email_from
        
        to_recipients = to_override if to_override is not None else self.config.email_to.split(',')
        cc_recipients = cc_override if cc_override is not None else self.config.email_cc.split(',')
        
        msg['To'] = ', '.join(filter(None, to_recipients))
        if cc_recipients and any(cc_recipients):
            msg['Cc'] = ', '.join(filter(None, cc_recipients))
        
        all_recipients = list(filter(None, to_recipients + cc_recipients))
        if not all_recipients:
            logger.warning("No recipients specified for email. Skipping send.")
            return False

        msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        # Attach plots to the email
        # First plot (auto-detect CID based on filename and available params)
        if plot_path and Path(plot_path).exists():
            try:
                with open(plot_path, 'rb') as f:
                    img_data = f.read()
                img = MIMEImage(img_data)

                filename_lower = Path(plot_path).name.lower()
                cid = None
                # Prefer per-CG CID if filename indicates per-CG
                if 'dl_per_cg' in filename_lower:
                    cid = body_params.get('dl_cg_plot')
                elif 'ul_per_cg' in filename_lower:
                    cid = body_params.get('ul_cg_plot')
                # Fallback to aggregated mapping
                if cid is None:
                    if 'ul_' in filename_lower or 'uplink' in filename_lower:
                        cid = body_params.get('performance_metrics')
                    else:
                        cid = body_params.get('traffic_trend')
                # Final fallback
                if cid is None:
                    cid = body_params.get('plot_cid', Path(plot_path).name)
                
                img.add_header('Content-ID', f"<{cid}>")
                img.add_header('Content-Disposition', 'inline', filename=Path(plot_path).name)
                msg.attach(img)
                logger.info(f"Plot 1 attached to email with CID {cid}.")
            except Exception as e:
                logger.error(f"Failed to attach first plot image {plot_path}: {e}")
        else:
            logger.warning(f"First plot path {plot_path} not found/valid.")
            
        # Second plot (auto-detect CID based on filename and available params)
        if plot_path2 and Path(plot_path2).exists():
            try:
                with open(plot_path2, 'rb') as f:
                    img_data = f.read()
                img = MIMEImage(img_data)

                filename_lower2 = Path(plot_path2).name.lower()
                cid2 = None
                # Prefer per-CG CID if filename indicates per-CG
                if 'dl_per_cg' in filename_lower2:
                    cid2 = body_params.get('dl_cg_plot')
                elif 'ul_per_cg' in filename_lower2:
                    cid2 = body_params.get('ul_cg_plot')
                # Fallback to aggregated mapping
                if cid2 is None:
                    if 'ul_' in filename_lower2 or 'uplink' in filename_lower2:
                        cid2 = body_params.get('performance_metrics', f"{body_params.get('plot_cid', 'plot_image')}_2")
                    else:
                        cid2 = body_params.get('traffic_trend', body_params.get('plot_cid', Path(plot_path2).name))
                # Final fallback
                if cid2 is None:
                    cid2 = f"{body_params.get('plot_cid', 'plot_image')}_2"

                img.add_header('Content-ID', f"<{cid2}>")
                img.add_header('Content-Disposition', 'inline', filename=Path(plot_path2).name)
                msg.attach(img)
                logger.info(f"Plot 2 attached to email with CID {cid2}.")
            except Exception as e:
                logger.error(f"Failed to attach second plot image {plot_path2}: {e}")
        elif plot_path2:
            logger.warning(f"Second plot path {plot_path2} not found/valid.")

        # Optional: attach per-CG plots if paths are provided in body_params
        try:
            extra_dl_cg_path = body_params.get('dl_cg_plot_path')
            if extra_dl_cg_path and Path(extra_dl_cg_path).exists():
                with open(extra_dl_cg_path, 'rb') as f:
                    img_data = f.read()
                img = MIMEImage(img_data)
                cid_extra_dl = body_params.get('dl_cg_plot', Path(extra_dl_cg_path).name)
                img.add_header('Content-ID', f"<{cid_extra_dl}>")
                img.add_header('Content-Disposition', 'inline', filename=Path(extra_dl_cg_path).name)
                msg.attach(img)
                logger.info(f"Per-CG DL plot attached with CID {cid_extra_dl}.")
        except Exception as e:
            logger.error(f"Failed to attach per-CG DL plot {body_params.get('dl_cg_plot_path')}: {e}")

        try:
            extra_ul_cg_path = body_params.get('ul_cg_plot_path')
            if extra_ul_cg_path and Path(extra_ul_cg_path).exists():
                with open(extra_ul_cg_path, 'rb') as f:
                    img_data = f.read()
                img = MIMEImage(img_data)
                cid_extra_ul = body_params.get('ul_cg_plot', Path(extra_ul_cg_path).name)
                img.add_header('Content-ID', f"<{cid_extra_ul}>")
                img.add_header('Content-Disposition', 'inline', filename=Path(extra_ul_cg_path).name)
                msg.attach(img)
                logger.info(f"Per-CG UL plot attached with CID {cid_extra_ul}.")
        except Exception as e:
            logger.error(f"Failed to attach per-CG UL plot {body_params.get('ul_cg_plot_path')}: {e}")

        import smtplib
        import ssl
        
        # First try with SSL
        ssl_success = False
        try:
            logger.info(f"Attempting to send email via SSL (port {self.config.smtp_port})...")
            with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port) as server:
                server.login(self.config.email_username, self.config.email_password)
                server.sendmail(self.config.email_from, all_recipients, msg.as_string())
            logger.info(f"✅ Email alert sent successfully via SSL to {', '.join(all_recipients)}.")
            ssl_success = True
            return True
        except ssl.SSLError as e:
            logger.warning(f"SSL error: {e}. Will try without SSL.")
        except ConnectionRefusedError as e:
            logger.warning(f"Connection refused on SSL port {self.config.smtp_port}: {e}. Will try without SSL.")
        except Exception as e:
            logger.warning(f"SSL email attempt failed: {e}. Will try without SSL.")
        
        # If SSL failed, try without SSL
        if not ssl_success:
            try:
                fallback_port = 587  # Common non-SSL SMTP port for STARTTLS
                logger.info(f"Attempting to send email without SSL (fallback to port {fallback_port})...")
                
                with smtplib.SMTP(self.config.smtp_host, fallback_port) as server:
                    server.ehlo()
                    if server.has_extn('STARTTLS'):
                        logger.info("STARTTLS available, upgrading connection...")
                        server.starttls()
                        server.ehlo()
                    
                    server.login(self.config.email_username, self.config.email_password)
                    server.sendmail(self.config.email_from, all_recipients, msg.as_string())
                
                logger.info(f"✅ Email alert sent successfully via non-SSL SMTP to {', '.join(all_recipients)}.")
                return True
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP Authentication Error: {e}. Check username/password.")
                return False
            except smtplib.SMTPServerDisconnected as e:
                logger.error(f"SMTP Server Disconnected: {e}. Check server address/port or network.")
                return False
            except smtplib.SMTPException as e:
                logger.error(f"SMTP error sending email: {e}")
                return False
            except Exception as e:
                logger.error(f"An unexpected error occurred sending email: {e}", exc_info=True)
                return False
        
        return ssl_success


class AnomalyWorkflow:
    """Orchestrates the anomaly detection workflow."""
    
    def __init__(self):
        """Initialize workflow components."""
        self.config = ConfigManager()
        self.db_service = DatabaseService(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.viz_service = VisualizationService(self.config)
        self.email_service = EmailService(self.config)
        self.processed_nodes_count = 0
        
    def run_rg_code_analysis(self, rg_code: str, test_email_recipient: Optional[str] = None) -> None:
        """Run anomaly detection and alerting for all traffic related to a specific RG code.
        
        This method aggregates traffic from all nodes with the same RG code and creates
        visualizations as specified in the requirements.
        
        Args:
            rg_code: The RG code to analyze (e.g., '70')
            test_email_recipient: Optional email recipient for test mode
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"{'🔬 ANALYZING RG CODE':^80}")
        logger.info(f"{'RG Code: ' + rg_code:^80}")
        logger.info("=" * 80)

        try:
            # 1. Get all nodes associated with this RG code
            logger.info(f"📊 Finding all nodes with RG code {rg_code}...")
            rg_nodes = self.db_service.get_nodes_by_rg_code(rg_code, hours_back=self.config.recent_hours * 2)
            
            if not rg_nodes:
                logger.warning(f"❌ No nodes found with RG code {rg_code}. Skipping analysis.")
                logger.info("=" * 80 + "\n")
                return
            
            logger.info(f"✅ Found {len(rg_nodes)} nodes with RG code {rg_code}")
            logger.info(f"   Sample nodes: {', '.join(rg_nodes[:3])}{'...' if len(rg_nodes) > 3 else ''}")
            
            # 2. Get aggregated data for all nodes with this RG code
            logger.info(f"📊 Fetching aggregated historical data for the past {self.config.historical_days} days (CG nodes for RG {rg_code})...")
            df_cg_agg = self.db_service.get_aggregated_cg_by_rg_code(rg_code, self.config.historical_days)
            
            if df_cg_agg.empty:
                logger.warning(f"❌ No historical CG data found for RG code {rg_code}. Skipping analysis.")
                logger.info("=" * 80 + "\n")
                return
                
            logger.info(f"✅ Retrieved {len(df_cg_agg)} aggregated data points (CG-only)")
            
            # Optional: skip RG code analysis for zero/low traffic aggregates
            avg_gb_72h = None
            if self.config.low_traffic_filter_enabled:
                try:
                    now_ts = datetime.now()
                    # Anchor to previous midnight; ignore today's partials in mean calcs
                    anchor_end = now_ts.replace(hour=0, minute=0, second=0, microsecond=0)
                    # Use configured window for low-traffic evaluation
                    window_start = anchor_end - timedelta(hours=self.config.mean_traffic_range_hours)
                    df_window = df_cg_agg[(df_cg_agg['timestamp'] >= window_start) & (df_cg_agg['timestamp'] < anchor_end)]
                    if df_window.empty:
                        df_window = df_cg_agg[(df_cg_agg['timestamp'] < anchor_end)]

                    total_counts = (df_window['rg_downlink'].fillna(0) + df_window['rg_uplink'].fillna(0))
                    # Compute hourly mean over the window based on actual time span
                    if len(df_window) > 1:
                        t_start = pd.to_datetime(df_window['timestamp'].min())
                        t_end = pd.to_datetime(df_window['timestamp'].max())
                        hours_span = max(1e-6, (t_end - t_start).total_seconds() / 3600.0)
                    else:
                        hours_span = 1.0
                    total_counts_sum = float(total_counts.sum())
                    avg_count_per_hour = total_counts_sum / hours_span
                    avg_gb_per_hour = (avg_count_per_hour * self.config.traffic_bytes_per_count) / (1024 ** 3)
                    avg_gb_72h = avg_gb_per_hour

                    if avg_gb_per_hour == 0:
                        logger.info(
                            f"🛑 Skipping RG code {rg_code}: aggregated hourly mean traffic over last {self.config.mean_traffic_range_hours}h (ending 00:00 today) is zero."
                        )
                        # If counts exist but GB/h is zero, likely missing bytes-per-count scaling
                        if total_counts_sum > 0:
                            logger.warning(
                                f"Low-traffic calc hint for RG {rg_code}: points={len(df_window)}, span_h={hours_span:.2f}, "
                                f"sum_counts={total_counts_sum:.2f}, bytes_per_count={self.config.traffic_bytes_per_count}. "
                                f"Consider setting TRAFFIC_BYTES_PER_COUNT or LOW_TRAFFIC_UNITS (kb|mb|gb)."
                            )
                        return

                    if avg_gb_per_hour < self.config.low_traffic_threshold_gb_per_hour:
                        logger.info(
                            f"🛑 Skipping RG code {rg_code}: hourly mean traffic over last {self.config.mean_traffic_range_hours}h (ending 00:00 today) is {avg_gb_per_hour:.6f} GB/h, "
                            f"below threshold {self.config.low_traffic_threshold_gb_per_hour} GB/h."
                        )
                        logger.debug(
                            f"Low-traffic details for RG {rg_code}: points={len(df_window)}, span_h={hours_span:.2f}, "
                            f"sum_counts={total_counts_sum:.2f}, bytes_per_count={self.config.traffic_bytes_per_count}, "
                            f"avg_count_per_hour={avg_count_per_hour:.2f}, avg_gb_per_hour={avg_gb_per_hour:.6f}"
                        )
                        return
                except Exception as e:
                    logger.warning(f"Low-traffic filter check failed for RG code {rg_code}: {e}")
            
            # 3. Get total traffic data across all RGs for comparison
            logger.info(f"📊 Fetching total traffic data across all RGs for comparison...")
            all_rg_nodes = self.db_service.get_distinct_rg_nodes(hours_back=self.config.recent_hours * 2)
            df_total = self.db_service.get_aggregated_rg_data(all_rg_nodes, self.config.historical_days)
            
            if df_total.empty:
                logger.warning(f"❌ No total traffic data available. Continuing with RG-specific analysis only.")
            else:
                logger.info(f"✅ Retrieved {len(df_total)} total traffic data points")
                
            # 4. Detect anomalies on total DL and UL traffic using the combined method
            logger.info(f"🔍 Running anomaly detection on both downlink and uplink traffic...")
            # Detect anomalies on CG-only aggregated data for this RG code
            detection_result = self.anomaly_detector.detect_anomalies_dl_ul(df_cg_agg)
            
            if not detection_result['success']:
                logger.error(f"❌ Anomaly detection failed: {detection_result['error']}")
                logger.info("=" * 80 + "\n")
                return
                
            df_total_with_anomalies = detection_result['data_with_anomalies']
            # Apply increment/decrement policy
            df_total_with_anomalies = self._apply_increment_decrement_policy(rg_code, df_total_with_anomalies)
 
            # Count recent anomalies
            recent_anomalies = df_total_with_anomalies[
                (df_total_with_anomalies['timestamp'] >= datetime.now() - timedelta(hours=self.config.recent_hours))
            ]
             
            recent_dl_anomalies = recent_anomalies[recent_anomalies['dl_anomaly'] == True]
            num_recent_dl_anomalies = len(recent_dl_anomalies)
             
            recent_ul_anomalies = recent_anomalies[recent_anomalies['ul_anomaly'] == True]
            num_recent_ul_anomalies = len(recent_ul_anomalies)
             
            # Total anomalies
            num_recent_anomalies = num_recent_dl_anomalies + num_recent_ul_anomalies
 
            # Mid-traffic zero warning: if 72h mean is in [LOW, MID) and last K hours are zero traffic
            mid_zero_trigger = False
            try:
                if avg_gb_72h is not None and \
                   (avg_gb_72h >= self.config.low_traffic_threshold_gb_per_hour) and \
                   (avg_gb_72h < self.config.mid_traffic_threshold_gb_per_hour):
                    recent_zero_window = datetime.now() - timedelta(hours=self.config.mid_waiting_hour_for_zero_traffic)
                    df_recent_zero = df_cg_agg[df_cg_agg['timestamp'] >= recent_zero_window]
                    if not df_recent_zero.empty:
                        zero_sum = float((df_recent_zero['rg_downlink'].fillna(0) + df_recent_zero['rg_uplink'].fillna(0)).sum())
                        if zero_sum == 0.0:
                            mid_zero_trigger = True
                            logger.info(
                                f"⚠️ Zero-traffic condition met for RG {rg_code}: last {self.config.mid_waiting_hour_for_zero_traffic}h zero traffic "
                                f"with mean {avg_gb_72h:.2f} GB/h over last {self.config.mean_traffic_range_hours}h between LOW and MID thresholds."
                            )
            except Exception as e:
                logger.warning(f"Mid-traffic zero check failed for RG {rg_code}: {e}")
 
            # Daily trend decrement trigger: if mean window is in [LOW, MID) and last 5 days daily means are strictly decreasing
            trend_decline_trigger = False
            try:
                if avg_gb_72h is not None and \
                   (avg_gb_72h >= self.config.low_traffic_threshold_gb_per_hour) and \
                   (avg_gb_72h < self.config.mid_traffic_threshold_gb_per_hour):
                    df_day = df_cg_agg.copy()
                    df_day['date'] = pd.to_datetime(df_day['timestamp']).dt.floor('D')
                    # Exclude today
                    anchor_end_date = datetime.now().date()
                    df_day = df_day[df_day['date'] < anchor_end_date]
                    # Combined daily mean
                    if 'total_rg' in df_day.columns:
                        daily_means = df_day.groupby('date')['total_rg'].mean().sort_index()
                    else:
                        daily_means = (df_day.groupby('date')[['rg_downlink','rg_uplink']].sum().sum(axis=1) / 
                                       df_day.groupby('date').size())
                    n_mid = self.config.mid_waiting_day_for_dec_trend_traffic
                    if len(daily_means) >= n_mid:
                        lastn_mid = daily_means.tail(n_mid).values
                        if all(lastn_mid[i] > lastn_mid[i+1] for i in range(0, n_mid - 1)):
                            trend_decline_trigger = True
                            logger.info(
                                f"⚠️ Daily trend decrement detected for RG {rg_code}: last {n_mid} daily means strictly decreasing."
                            )
            except Exception as e:
                logger.warning(f"Daily trend decrement check failed for RG {rg_code}: {e}")
 
            # High-traffic zero warning: if mean > MID and last H hours are zero traffic
            high_zero_trigger = False
            try:
                if avg_gb_72h is not None and (avg_gb_72h > self.config.mid_traffic_threshold_gb_per_hour):
                    recent_zero_window_h = datetime.now() - timedelta(hours=self.config.high_waiting_hour_for_zero_traffic)
                    df_recent_zero_h = df_cg_agg[df_cg_agg['timestamp'] >= recent_zero_window_h]
                    if not df_recent_zero_h.empty:
                        zero_sum_h = float((df_recent_zero_h['rg_downlink'].fillna(0) + df_recent_zero_h['rg_uplink'].fillna(0)).sum())
                        if zero_sum_h == 0.0:
                            high_zero_trigger = True
                            logger.info(
                                f"⚠️ High-tier zero-traffic condition for RG {rg_code}: last {self.config.high_waiting_hour_for_zero_traffic}h zero traffic "
                                f"with mean {avg_gb_72h:.2f} GB/h over last {self.config.mean_traffic_range_hours}h (> MID)."
                            )
            except Exception as e:
                logger.warning(f"High-traffic zero check failed for RG {rg_code}: {e}")
 
            # High-traffic daily decrement: if mean > MID and last N days decreasing
            high_trend_decline_trigger = False
            try:
                if avg_gb_72h is not None and (avg_gb_72h > self.config.mid_traffic_threshold_gb_per_hour):
                    df_day_h = df_cg_agg.copy()
                    df_day_h['date'] = pd.to_datetime(df_day_h['timestamp']).dt.floor('D')
                    # Exclude today
                    anchor_end_date_h = datetime.now().date()
                    df_day_h = df_day_h[df_day_h['date'] < anchor_end_date_h]
                    if 'total_rg' in df_day_h.columns:
                        daily_means_h = df_day_h.groupby('date')['total_rg'].mean().sort_index()
                    else:
                        daily_means_h = (df_day_h.groupby('date')[['rg_downlink','rg_uplink']].sum().sum(axis=1) / 
                                         df_day_h.groupby('date').size())
                    if len(daily_means_h) >= self.config.high_waiting_day_for_dec_trend_traffic:
                        n = self.config.high_waiting_day_for_dec_trend_traffic
                        lastn = daily_means_h.tail(n).values
                        if all(lastn[i] > lastn[i+1] for i in range(0, n-1)):
                            high_trend_decline_trigger = True
                            logger.info(
                                f"⚠️ High-tier daily trend decrement for RG {rg_code}: last {n} daily means strictly decreasing."
                            )
            except Exception as e:
                logger.warning(f"High-traffic daily trend check failed for RG {rg_code}: {e}")
 
             
            # Summarized results
            logger.info(f"\n{'ANALYSIS RESULTS':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Time window: Last {self.config.recent_hours} hours | Alert threshold: > {self.config.min_anomalies_for_alert}")
            logger.info(f"DL anomalies detected: {num_recent_dl_anomalies}")
            logger.info(f"UL anomalies detected: {num_recent_ul_anomalies}")
            logger.info(f"Total anomalies detected: {num_recent_anomalies}")
             
            # Determine if alert is needed
            alert_needed = (
                (num_recent_anomalies > self.config.min_anomalies_for_alert)
                or mid_zero_trigger or trend_decline_trigger
                or high_zero_trigger or high_trend_decline_trigger
            )
             
            # Determine issue type and reason
            issue_type = "Anomaly"
            issue_reason = "Recent anomalies exceed threshold"
            if high_zero_trigger:
                issue_type = "High-Tier Zero Traffic"
                issue_reason = f"0 traffic for {self.config.high_waiting_hour_for_zero_traffic}h; mean > MID"
            elif high_trend_decline_trigger:
                issue_type = "High-Tier Decreasing Trend"
                issue_reason = f"Daily mean decreasing {self.config.high_waiting_day_for_dec_trend_traffic}d; mean > MID"
            elif mid_zero_trigger:
                issue_type = "Mid-Tier Zero Traffic"
                issue_reason = f"0 traffic for {self.config.mid_waiting_hour_for_zero_traffic}h; mean in [LOW,MID)"
            elif trend_decline_trigger:
                issue_type = "Mid-Tier Decreasing Trend"
                issue_reason = f"Daily mean decreasing {self.config.mid_waiting_day_for_dec_trend_traffic}d; mean in [LOW,MID)"
            elif num_recent_anomalies > self.config.min_anomalies_for_alert:
                issue_type = "Model Anomaly"
                issue_reason = f"DL/UL anomalies > {self.config.min_anomalies_for_alert} in last {self.config.recent_hours}h"
             
            # 5. Generate visualizations only if alert is needed
            plot_paths_36h = None
            plot_paths_30d = None
            if alert_needed:
                logger.info(f"\n📈 Generating RG code traffic analysis visualization...")
                
                                # Create separate DFs for RG code DL and UL (aggregated over CG nodes)
                df_rg_dl = df_cg_agg[['timestamp', 'rg_downlink']].copy()
                df_rg_ul = df_cg_agg[['timestamp', 'rg_uplink']].copy()
                
                # Compute mean DL and UL over configured window
                try:
                    # Anchor to previous midnight; ignore today
                    end_time_mean = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    start_time_mean = end_time_mean - timedelta(hours=self.config.mean_traffic_range_hours)
                    df_mean = df_cg_agg[(df_cg_agg['timestamp'] >= start_time_mean) & (df_cg_agg['timestamp'] < end_time_mean)]
                    if df_mean.empty:
                        df_mean = df_cg_agg[(df_cg_agg['timestamp'] < end_time_mean)]
                    # Time-span based hourly mean
                    if len(df_mean) > 1:
                        t_start_m = pd.to_datetime(df_mean['timestamp'].min())
                        t_end_m = pd.to_datetime(df_mean['timestamp'].max())
                        hours_span_m = max(1e-6, (t_end_m - t_start_m).total_seconds() / 3600.0)
                    else:
                        hours_span_m = 1.0
                    mean_dl_per_h = float(df_mean['rg_downlink'].sum()) / hours_span_m
                    mean_ul_per_h = float(df_mean['rg_uplink'].sum()) / hours_span_m
                    # Convert to GB/h if counts are bytes per count configured
                    mean_dl_gb_h = (mean_dl_per_h * self.config.traffic_bytes_per_count) / (1024 ** 3)
                    mean_ul_gb_h = (mean_ul_per_h * self.config.traffic_bytes_per_count) / (1024 ** 3)
                    mean_dl_ul_72h_str = f"DL: {mean_dl_gb_h:.2f}\n || UL: {mean_ul_gb_h:.2f}"
                except Exception as e:
                    logger.warning(f"Could not compute 72h mean traffic: {e}")
                    mean_dl_ul_72h_str = "N/A"
                
                # Get data for per-CG traffic
                logger.info(f"📊 Fetching per-CG traffic data...")
                # Find CG nodes that belong to this RG code by suffix
                cg_nodes = self.db_service.get_cg_nodes_for_rg_code(rg_code, hours_back=self.config.recent_hours * 2)
                
                # Create DataFrames for per-CG traffic
                df_per_cg_dl = None
                df_per_cg_ul = None
                
                if cg_nodes:
                    logger.info(f"✅ Found {len(cg_nodes)} associated CG nodes")
                    logger.info(f"   Sample CG nodes: {', '.join(cg_nodes[:3])}{'...' if len(cg_nodes) > 3 else ''}")
                    
                    # Fetch per-CG traffic data
                    logger.info(f"📊 Fetching traffic data from RG code {rg_code} to each CG node...")
                    cg_traffic = self.db_service.get_rg_to_cg_traffic(
                    rg_code=rg_code,
                        cg_nodes=cg_nodes,
                        days_back=self.config.historical_days
                    )
                    
                    df_per_cg_dl = cg_traffic['dl']
                    df_per_cg_ul = cg_traffic['ul']
                    
                    if not df_per_cg_dl.empty and not df_per_cg_ul.empty:
                        logger.info(f"✅ Retrieved traffic data for {len(df_per_cg_dl['cg_node'].unique())} CG nodes")
                    else:
                        logger.warning(f"⚠️ No per-CG traffic data found for RG code {rg_code}")
                
                # Generate visualization for 30 days only
                plot_paths = self.viz_service.plot_rg_code_analysis(
                    rg_code=rg_code,
                    df_total=df_total,
                    df_total_with_anomalies=df_total_with_anomalies,
                    df_rg_dl=df_rg_dl,
                    df_rg_ul=df_rg_ul,
                    time_range='30 Days',
                    df_per_cg_dl=df_per_cg_dl,
                    df_per_cg_ul=df_per_cg_ul,
                    mid_trend_days=(self.config.mid_waiting_day_for_dec_trend_traffic if trend_decline_trigger else None),
                    high_trend_days=(self.config.high_waiting_day_for_dec_trend_traffic if high_trend_decline_trigger else None)
                )
                if plot_paths:
                    logger.info(f"✅ 30-Day RG code analysis plots saved")
                    for plot_type, path in plot_paths.items():
                        logger.info(f"  - {plot_type}: {path}")
                
            # 6. Send email alert if significant anomalies found
            logger.info(f"\n{'ALERT STATUS':^80}")
            logger.info(f"{'-' * 40:^80}")
            
            if alert_needed:
                severity = "CRITICAL" if num_recent_anomalies >= self.config.min_anomalies_for_alert * 2 else "WARNING"
                severity_icon = "🔴" if severity == "CRITICAL" else "🟠"
                
                logger.info(f"{severity_icon} {severity} ALERT will be sent for RG CODE {rg_code}")
                logger.info(f"📧 Recipients: {test_email_recipient if test_email_recipient else self.config.email_to}")
                
                alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                summary = f"{num_recent_anomalies} anomalies detected for RG code {rg_code} traffic (DL: {num_recent_dl_anomalies}, UL: {num_recent_ul_anomalies}) in the last {self.config.recent_hours} hours."
                
                additional_details_parts = [
                    f"<b>RG Code Analysis:</b> {rg_code}",
                    f"<b>Nodes Included:</b> {', '.join(rg_nodes[:5])}{'...' if len(rg_nodes) > 5 else ''} ({len(rg_nodes)} total)",
                    f"<b>Detection Period:</b> Last {self.config.recent_hours} hours.",
                    f"<b>DL Anomalies:</b> {num_recent_dl_anomalies}",
                    f"<b>UL Anomalies:</b> {num_recent_ul_anomalies}",
                    f"<b>Total Anomalies:</b> {num_recent_anomalies}",
                    f"<b>Sensitivity Threshold:</b> {self.config.sensitivity}",
                    f"<b>Minimum Anomalies for Alert:</b> {self.config.min_anomalies_for_alert}"
                ]
                if avg_gb_72h is not None:
                    additional_details_parts.append(
                        f"<b>Mean {self.config.mean_traffic_range_hours}h Traffic (GB/h):</b> {avg_gb_72h:.2f} (combined)"
                    )
                if mid_zero_trigger:
                    additional_details_parts.append(
                        f"<b>Zero Traffic Condition:</b> Observed 0 traffic for last {self.config.mid_waiting_hour_for_zero_traffic} hours "
                        f"with {self.config.mean_traffic_range_hours}h mean in [{self.config.low_traffic_threshold_gb_per_hour}, {self.config.mid_traffic_threshold_gb_per_hour})."
                    )
                if trend_decline_trigger:
                    additional_details_parts.append(
                        f"<b>Daily Trend:</b> Last {self.config.mid_waiting_day_for_dec_trend_traffic} daily means strictly decreasing while {self.config.mean_traffic_range_hours}h mean is between LOW and MID thresholds."
                    )
                if high_zero_trigger:
                    additional_details_parts.append(
                        f"<b>High-tier Zero Traffic:</b> Observed 0 traffic for last {self.config.high_waiting_hour_for_zero_traffic} hours "
                        f"with {self.config.mean_traffic_range_hours}h mean > {self.config.mid_traffic_threshold_gb_per_hour} GB/h."
                    )
                if high_trend_decline_trigger:
                    additional_details_parts.append(
                        f"<b>High-tier Daily Trend:</b> Last {self.config.high_waiting_day_for_dec_trend_traffic} daily means strictly decreasing "
                        f"with {self.config.mean_traffic_range_hours}h mean > {self.config.mid_traffic_threshold_gb_per_hour} GB/h."
                    )
                if num_recent_dl_anomalies > 0:
                    additional_details_parts.append("<br><b>Recent DL Anomaly Values (Timestamp, Actual):</b>")
                    for _, row in recent_dl_anomalies.head(3).iterrows():
                        try:
                            if 'dl_yhat_lower' in row and 'dl_yhat_upper' in row:
                                additional_details_parts.append(
                                    f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_downlink']:.2f} (Expected Range: {row['dl_yhat_lower']:.2f} - {row['dl_yhat_upper']:.2f})"
                                )
                            else:
                                additional_details_parts.append(
                                    f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_downlink']:.2f}"
                                )
                        except KeyError:
                            # Fallback if any expected keys are missing
                            additional_details_parts.append(
                                f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_downlink']:.2f}"
                        )
                if num_recent_ul_anomalies > 0:
                    additional_details_parts.append("<br><b>Recent UL Anomaly Values (Timestamp, Actual):</b>")
                    for _, row in recent_ul_anomalies.head(3).iterrows():
                        try:
                            if 'ul_yhat_lower' in row and 'ul_yhat_upper' in row:
                                additional_details_parts.append(
                                    f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_uplink']:.2f} (Expected Range: {row['ul_yhat_lower']:.2f} - {row['ul_yhat_upper']:.2f})"
                                )
                            else:
                                additional_details_parts.append(
                                    f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_uplink']:.2f}"
                                )
                        except KeyError:
                            # Fallback if any expected keys are missing
                            additional_details_parts.append(
                                f"- {row['timestamp'].strftime('%H:%M')} : {row['rg_uplink']:.2f}"
                        )
                additional_details = "<br>".join(additional_details_parts)

                email_subject = f"Core - RG Code {rg_code}: {issue_type} - ({issue_reason})"
                
                # Choose plots to embed
                email_plot_paths = []
                if plot_paths:
                    # Add the plots for DL and UL aggregated traffic with anomaly detection
                    if 'dl_agg' in plot_paths:
                        email_plot_paths.append(plot_paths['dl_agg'])
                    if 'ul_agg' in plot_paths:
                        email_plot_paths.append(plot_paths['ul_agg'])
                    # Add the plots for per-CG traffic
                    if 'dl_cg' in plot_paths:
                        email_plot_paths.append(plot_paths['dl_cg'])
                    if 'ul_cg' in plot_paths:
                        email_plot_paths.append(plot_paths['ul_cg'])
                
                body_params = {
                    'node_name': f"RG Code {rg_code}",
                    'kpi_name': f"Traffic Analysis (DL & UL)",
                    'alert_time': alert_time,
                    'severity': severity,
                    'summary': summary,
                    'additional_details': additional_details,
                    'system_id': self.config.system_id,
                    'num_recent_anomalies': str(num_recent_anomalies),
                    'detection_period': f"{self.config.recent_hours} hours",
                    'ticket_number': f"RG-{datetime.now().strftime('%Y')}-{rg_code}-{str(hash(alert_time) % 10000).zfill(4)}",
                    'plot_cid': 'rg_code_plot_cid',
                    # New CIDs for extra images
                    'dl_cg_plot': 'rg_code_dl_cg_plot',
                    'ul_cg_plot': 'rg_code_ul_cg_plot',
                    'traffic_trend': 'rg_code_dl_plot',
                    'performance_metrics': 'rg_code_ul_plot',
                    # Provide aggregated RG+CG list for template
                    'rg_cg_list': ", ".join(sorted(set((rg_nodes or []) + ((locals().get('cg_nodes') or [])))))[:800],
                    # 72h mean traffic string for table
                    'mean_dl_ul_72h': mean_dl_ul_72h_str,
                    'mean_range_hours': str(self.config.mean_traffic_range_hours),
                    'issue_type': issue_type,
                    'issue_reason': issue_reason,
                }

                to_recipients = [test_email_recipient] if test_email_recipient else None
                cc_recipients = [] if test_email_recipient else None

                # Prepare email parameters for the compatible template
                if len(email_plot_paths) >= 2:
                    # We have at least 2 plots (DL and UL) to include in the email
                    dl_plot = email_plot_paths[0] if 'dl_agg' in plot_paths else None
                    ul_plot = email_plot_paths[1] if 'ul_agg' in plot_paths else None

                    # Enhanced parameters for the compatible template
                    enhanced_params = {
                        **body_params,
                        'interface': f"RG Code {rg_code}",
                        'detected_time': alert_time,
                        'anomaly_type': issue_type,
                        'ticket_link': '#',
                        'current_traffic': str(num_recent_anomalies),
                        'threshold_exceeded': str(self.config.min_anomalies_for_alert),
                        'duration': f"{self.config.recent_hours}h",
                        'affected_users': f"{self.config.historical_days}d",
                        'current_traffic_val': f"{num_recent_dl_anomalies} DL, {num_recent_ul_anomalies} UL",
                        'baseline': "Normal",
                        'threshold': str(self.config.min_anomalies_for_alert),
                        'utilization': f"{num_recent_anomalies}/{self.config.min_anomalies_for_alert}",
                        'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        # New: include aggregated RG-CG list for display in the email header summary
                        'rg_cg_list': ", ".join(sorted(set((rg_nodes or []) + (self.db_service.get_cg_nodes_for_rg_code(rg_code, hours_back=self.config.recent_hours * 2) or []))))[:800]
                    }

                    # Also include per-CG plot paths so they can be embedded in the same email
                    if plot_paths.get('dl_cg'):
                        enhanced_params['dl_cg_plot_path'] = plot_paths.get('dl_cg')
                    if plot_paths.get('ul_cg'):
                        enhanced_params['ul_cg_plot_path'] = plot_paths.get('ul_cg')

                    # Send email with the main DL/UL plots embedded (and per-CG if provided)
                    email_result = self.email_service.send_alert_email(
                        email_subject,
                        enhanced_params,
                        dl_plot,  # First plot (DL aggregated)
                        ul_plot,  # Second plot (UL aggregated)
                        to_override=to_recipients,
                        cc_override=cc_recipients
                    )
                    logger.info(f"✉️ Email status: {'✅ Sent' if email_result else '❌ Failed'}")

                    # Send a follow-up email embedding per-CG plots when available
                    if 'dl_cg' in plot_paths or 'ul_cg' in plot_paths:
                        followup_params = {
                            **body_params,
                            'interface': f"RG Code {rg_code}",
                            'detected_time': alert_time,
                            'anomaly_type': issue_type,
                            'ticket_link': '#',
                            'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'rg_cg_list': ", ".join(sorted(set((rg_nodes or []) + (self.db_service.get_cg_nodes_for_rg_code(rg_code, hours_back=self.config.recent_hours * 2) or []))))[:800]
                        }
                        dl_cg_plot = plot_paths.get('dl_cg')
                        ul_cg_plot = plot_paths.get('ul_cg')
                        # Reuse the same API: first is dl per-CG, second is ul per-CG
                        self.email_service.send_alert_email(
                            f"[Per-CG] {email_subject}",
                            followup_params,
                            dl_cg_plot,
                            ul_cg_plot,
                            to_override=to_recipients,
                            cc_override=cc_recipients
                        )
                else:
                    # Fall back to sending a single plot if we don't have enough
                    if email_plot_paths:
                        email_result = self.email_service.send_alert_email(
                            email_subject, 
                            body_params, 
                            email_plot_paths[0],
                            to_override=to_recipients,
                            cc_override=cc_recipients
                        )
                        logger.info(f"✉️ Email status: {'✅ Sent' if email_result else '❌ Failed'}")
                    else:
                        logger.warning("⚠️ No plots available to send in email")
            else:
                logger.info(f"✅ No alert needed - {num_recent_anomalies} anomalies below threshold (> {self.config.min_anomalies_for_alert})")
            
            # Clean output/charts directory after analyzing each case
            self._clean_output_directory()
            
            self.processed_nodes_count += 1
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❗ Error processing RG code {rg_code}: {e}", exc_info=True)
            logger.info("=" * 80 + "\n")

    def _log_summary(self, start_time: datetime):
        """Logs a summary of the workflow execution."""
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info(f"{'📊 ANOMALY DETECTION WORKFLOW SUMMARY 📊':^80}")
        logger.info("=" * 80)
        
        # Format the timestamp and duration information
        logger.info(f"\n{'⏱️ EXECUTION TIME':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"{'🚀 Start time:':<30} {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'🏁 End time:':<30} {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'⌛ Total duration:':<30} {str(duration).split('.')[0]}")
        
        # Format the configuration information
        logger.info(f"\n{'⚙️ CONFIGURATION':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"{'📊 Nodes processed:':<30} {self.processed_nodes_count}")
        logger.info(f"{'🔍 Max nodes limit:':<30} {'All' if self.config.max_nodes == 0 else self.config.max_nodes}")
        logger.info(f"{'🎯 Sensitivity:':<30} {self.config.sensitivity}")
        logger.info(f"{'📆 Historical days:':<30} {self.config.historical_days}")
        logger.info(f"{'🚨 Min anomalies for alert:':<30} {self.config.min_anomalies_for_alert}")
        logger.info(f"{'⏰ Recent hours check:':<30} {self.config.recent_hours}")
        
        # Format the system status information
        logger.info(f"\n{'🔧 SYSTEM STATUS':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"{'📝 Log level:':<30} {logging.getLevelName(logger.getEffectiveLevel())}")
        logger.info(f"{'📈 Prophet available:':<30} {'✅ Yes' if PROPHET_AVAILABLE else '❌ No'}")
        logger.info(f"{'🗄️ Oracle DB available:':<30} {'✅ Yes' if ORACLE_AVAILABLE else '❌ No'}")
        
        template_status = '✅ Found' if self.email_service.email_template_path.exists() else '❌ Not found (using default)'
        logger.info(f"{'📧 Email template:':<30} {self.email_service.email_template_path.name} [{template_status}]")
        
        logger.info("\n" + "=" * 80)

    def run_rg_aggregated_analysis(self, rg_base: str, rg_nodes: List[str], cg_nodes: List[str], metric_column: str = 'total_rg', test_email_recipient: Optional[str] = None) -> None:
        """Run anomaly detection and alerting for aggregated RG traffic from all associated CG nodes."""
        logger.info("\n" + "=" * 80)
        logger.info(f"{'🔬 ANALYZING RG GROUP':^80}")
        logger.info(f"{'RG Base: ' + rg_base:^80}")
        logger.info(f"{'RG Nodes: ' + str(len(rg_nodes)) + ' nodes':^80}")
        logger.info(f"{'CG Nodes: ' + str(len(cg_nodes)) + ' nodes':^80}")
        logger.info(f"{'KPI: ' + metric_column:^80}")
        logger.info("=" * 80)

        try:
            # 1. Get aggregated historical data for all CG nodes in this RG group
            logger.info(f"📊 Fetching aggregated historical data for the past {self.config.historical_days} days...")
            logger.info(f"   - RG Nodes ({len(rg_nodes)}): {', '.join(rg_nodes[:3])}{'...' if len(rg_nodes) > 3 else ''}")
            logger.info(f"   - CG Nodes ({len(cg_nodes)}): {', '.join(cg_nodes[:3])}{'...' if len(cg_nodes) > 3 else ''}")
            
            all_nodes = rg_nodes + cg_nodes
            df_aggregated = self.db_service.get_aggregated_rg_data(all_nodes, self.config.historical_days)
            
            if df_aggregated.empty:
                logger.warning(f"❌ No aggregated historical data found for RG group {rg_base}. Skipping analysis.")
                logger.info("=" * 80 + "\n")
                return
            logger.info(f"✅ Retrieved {len(df_aggregated)} aggregated data points")

            # 2. Detect anomalies on aggregated data
            logger.info(f"🔍 Running anomaly detection on aggregated RG traffic (sensitivity: {self.config.sensitivity})...")
            detection_result = self.anomaly_detector.detect_anomalies(df_aggregated, metric_column)
            if not detection_result['success']:
                logger.error(f"❌ Anomaly detection failed: {detection_result['error']}")
                logger.info("=" * 80 + "\n")
                return
            
            df_with_anomalies = detection_result['data_with_anomalies']
            forecast_df = detection_result['forecast_df']
            model = detection_result['model']

            recent_anomalies = df_with_anomalies[
                (df_with_anomalies['timestamp'] >= datetime.now() - timedelta(hours=self.config.recent_hours)) &
                (df_with_anomalies['anomaly'] == True)
            ]
            num_recent_anomalies = len(recent_anomalies)
            
            logger.info(f"\n{'ANALYSIS RESULTS':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Time window: Last {self.config.recent_hours} hours | Alert threshold: > {self.config.min_anomalies_for_alert}")
            logger.info(f"Anomalies detected: {num_recent_anomalies}")
            
            alert_needed = num_recent_anomalies > self.config.min_anomalies_for_alert
            
            plot_paths_dict = None
            if alert_needed:
                if num_recent_anomalies > 0:
                    logger.info(f"\n{'ANOMALY DETAILS':^80}")
                    logger.info(f"{'-' * 40:^80}")
                    logger.info(f"First anomaly: {recent_anomalies['timestamp'].min().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    logger.info(f"Last anomaly:  {recent_anomalies['timestamp'].max().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    self._create_alarm_summary(rg_base, metric_column, num_recent_anomalies, recent_anomalies)
                
                logger.info(f"\n📈 Generating visualization for RG group alert...")
                plot_paths_dict = self.viz_service.plot_kpi_with_anomalies(df_with_anomalies, rg_base, metric_column, forecast_df)
                
                plot_path_30_days = plot_paths_dict.get('30_Days') if plot_paths_dict else None
                plot_path_36_hours = plot_paths_dict.get('36_Hours') if plot_paths_dict else None
                if plot_path_30_days:
                    logger.info(f"✅ 30-Day Plot saved to: {plot_path_30_days}")
                if plot_path_36_hours:
                    logger.info(f"✅ 36-Hour Plot saved to: {plot_path_36_hours}")
            
            logger.info(f"\n{'ALERT STATUS':^80}")
            logger.info(f"{'-' * 40:^80}")
            
            if alert_needed:
                severity = "CRITICAL" if num_recent_anomalies >= self.config.min_anomalies_for_alert * 2 else "WARNING"
                severity_icon = "🔴" if severity == "CRITICAL" else "🟠"
                
                logger.info(f"{severity_icon} {severity} ALERT will be sent for RG GROUP")
                logger.info(f"📧 Recipients: {test_email_recipient if test_email_recipient else self.config.email_to}")
                
                alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                rg_nodes_summary = f"{len(rg_nodes)} RG nodes" if len(rg_nodes) > 1 else f"RG node {rg_nodes[0]}"
                cg_nodes_summary = f"{len(cg_nodes)} CG nodes" if len(cg_nodes) > 1 else f"CG node {cg_nodes[0]}"
                
                summary = f"{num_recent_anomalies} anomalies detected for aggregated KPI '{metric_column if metric_column != 'total_rg' else 'TOTAL SG/RG'}' across RG group '{rg_base}' ({rg_nodes_summary} + {cg_nodes_summary}) in the last {self.config.recent_hours} hours."
                first_anomaly_time = recent_anomalies['timestamp'].min().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else "N/A"
                last_anomaly_time = recent_anomalies['timestamp'].max().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else "N/A"

                additional_details_parts = [
                    f"<b>RG Group Analysis:</b> {rg_base}",
                    f"<b>RG Nodes Included:</b> {', '.join(rg_nodes[:5])}{'...' if len(rg_nodes) > 5 else ''} ({len(rg_nodes)} total)",
                    f"<b>CG Nodes Included:</b> {', '.join(cg_nodes[:5])}{'...' if len(cg_nodes) > 5 else ''} ({len(cg_nodes)} total)",
                    f"<b>Detection Period:</b> Last {self.config.recent_hours} hours.",
                    f"<b>Number of Anomalies:</b> {num_recent_anomalies}",
                    f"<b>First Anomaly (in period):</b> {first_anomaly_time}",
                    f"<b>Last Anomaly (in period):</b> {last_anomaly_time}",
                    f"<b>Sensitivity Threshold:</b> {self.config.sensitivity}",
                    f"<b>Minimum Anomalies for Alert:</b> {self.config.min_anomalies_for_alert}"
                ]
                if not recent_anomalies.empty:
                    additional_details_parts.append("<br><b>Recent Anomaly Values (Timestamp, Actual, Forecasted Low, Forecasted High):</b>")
                    for _, row in recent_anomalies.head(5).iterrows():
                        additional_details_parts.append(
                            f"- {row['timestamp'].strftime('%H:%M')} : {row[metric_column]:.2f} (Expected Range: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})"
                        )
                additional_details = "<br>".join(additional_details_parts)

                email_subject = f"Core - RG Code {rg_code}: {issue_type} - ({issue_reason})"
                
                # Choose a plot to embed (prefer 36h)
                chosen_plot_path = (plot_paths_dict or {}).get('36_Hours') if plot_paths_dict else None
                if not chosen_plot_path:
                    chosen_plot_path = (plot_paths_dict or {}).get('30_Days') if plot_paths_dict else None
                
                body_params = {
                    'node_name': f"{rg_base} (RG Group)",
                    'kpi_name': f"{metric_column if metric_column != 'total_rg' else 'TOTAL SG/RG'} - Aggregated",
                    'alert_time': alert_time,
                    'severity': severity,
                    'summary': summary,
                    'additional_details': additional_details,
                    'system_id': self.config.system_id,
                    'num_recent_anomalies': str(num_recent_anomalies),
                    'detection_period': f"{self.config.recent_hours} hours",
                    'ticket_number': f"RG-{datetime.now().strftime('%Y')}-{str(hash(rg_base + alert_time) % 100000).zfill(6)}",
                    'plot_cid': 'rg_group_plot_cid',
                }

                to_recipients = [test_email_recipient] if test_email_recipient else None
                cc_recipients = [] if test_email_recipient else None

                email_result = self.email_service.send_alert_email(
                    email_subject, 
                    body_params, 
                    chosen_plot_path,
                    to_override=to_recipients,
                    cc_override=cc_recipients
                )
                logger.info(f"✉️ Email status: {'✅ Sent' if email_result else '❌ Failed'}")
            else:
                logger.info(f"✅ No alert needed for RG group - {num_recent_anomalies} anomalies below threshold (> {self.config.min_anomalies_for_alert})")
            
            self._clean_output_directory()
            
            self.processed_nodes_count += 1
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❗ Error processing RG group {rg_base}: {e}", exc_info=True)
            logger.info("=" * 80 + "\n")

    def run_node_analysis(self, node_name: str, metric_column: str = 'total_rg', test_email_recipient: Optional[str] = None) -> None:
        """Run anomaly detection and alerting for a single node."""
        logger.info("\n" + "=" * 80)
        logger.info(f"{'🔬 ANALYZING NODE':^80}")
        logger.info(f"{'Node: ' + node_name:^80}")
        logger.info(f"{'KPI: ' + metric_column:^80}")
        logger.info("=" * 80)

        try:
            # 1. Get historical data
            logger.info(f"📊 Fetching historical data for the past {self.config.historical_days} days...")
            df_node = self.db_service.get_node_historical_data(node_name, self.config.historical_days)
            if df_node.empty:
                logger.warning(f"❌ No historical data found for node {node_name}. Skipping analysis.")
                logger.info("=" * 80 + "\n")
                return
            logger.info(f"✅ Retrieved {len(df_node)} data points")

            # 2. Detect anomalies
            logger.info(f"🔍 Running anomaly detection (sensitivity: {self.config.sensitivity})...")
            detection_result = self.anomaly_detector.detect_anomalies(df_node, metric_column)
            if not detection_result['success']:
                logger.error(f"❌ Anomaly detection failed: {detection_result['error']}")
                logger.info("=" * 80 + "\n")
                return
            
            df_with_anomalies = detection_result['data_with_anomalies']
            forecast_df = detection_result['forecast_df']
            model = detection_result['model']

            recent_anomalies = df_with_anomalies[
                (df_with_anomalies['timestamp'] >= datetime.now() - timedelta(hours=self.config.recent_hours)) &
                (df_with_anomalies['anomaly'] == True)
            ]
            num_recent_anomalies = len(recent_anomalies)
            
            logger.info(f"\n{'ANALYSIS RESULTS':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Time window: Last {self.config.recent_hours} hours | Alert threshold: > {self.config.min_anomalies_for_alert}")
            logger.info(f"Anomalies detected: {num_recent_anomalies}")
            
            alert_needed = num_recent_anomalies > self.config.min_anomalies_for_alert
            
            plot_paths_dict = None
            if alert_needed:
                if num_recent_anomalies > 0:
                    logger.info(f"\n{'ANOMALY DETAILS':^80}")
                    logger.info(f"{'-' * 40:^80}")
                    logger.info(f"First anomaly: {recent_anomalies['timestamp'].min().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    logger.info(f"Last anomaly:  {recent_anomalies['timestamp'].max().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    self._create_alarm_summary(node_name, metric_column, num_recent_anomalies, recent_anomalies)
                
                logger.info(f"\n📈 Generating visualization for alert...")
                plot_paths_dict = self.viz_service.plot_kpi_with_anomalies(df_with_anomalies, node_name, metric_column, forecast_df)
                
                plot_path_30_days = plot_paths_dict.get('30_Days')
                plot_path_36_hours = plot_paths_dict.get('36_Hours')
                if plot_path_30_days:
                    logger.info(f"✅ 30-Day Plot saved to: {plot_path_30_days}")
                if plot_path_36_hours:
                    logger.info(f"✅ 36-Hour Plot saved to: {plot_path_36_hours}")
            
            logger.info(f"\n{'ALERT STATUS':^80}")
            logger.info(f"{'-' * 40:^80}")
            
            if alert_needed:
                severity = "CRITICAL" if num_recent_anomalies >= self.config.min_anomalies_for_alert * 2 else "WARNING"
                severity_icon = "🔴" if severity == "CRITICAL" else "🟠"
                
                logger.info(f"{severity_icon} {severity} ALERT will be sent")
                logger.info(f"📧 Recipients: {test_email_recipient if test_email_recipient else self.config.email_to}")
                
                alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                summary = f"{num_recent_anomalies} anomalies detected for KPI '{metric_column if metric_column != 'total_rg' else 'TOTAL SG/RG'}' on node '{node_name}' in the last {self.config.recent_hours} hours."
                first_anomaly_time = recent_anomalies['timestamp'].min().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else "N/A"
                last_anomaly_time = recent_anomalies['timestamp'].max().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else "N/A"

                additional_details_parts = [
                    f"<b>Detection Period:</b> Last {self.config.recent_hours} hours.",
                    f"<b>Number of Anomalies:</b> {num_recent_anomalies}",
                    f"<b>First Anomaly (in period):</b> {first_anomaly_time}",
                    f"<b>Last Anomaly (in period):</b> {last_anomaly_time}",
                    f"<b>Sensitivity Threshold:</b> {self.config.sensitivity}",
                    f"<b>Minimum Anomalies for Alert:</b> {self.config.min_anomalies_for_alert}"
                ]
                if not recent_anomalies.empty:
                    additional_details_parts.append("<br><b>Recent Anomaly Values (Timestamp, Actual, Forecasted Low, Forecasted High):</b>")
                    for _, row in recent_anomalies.head(5).iterrows():
                        additional_details_parts.append(
                            f"- {row['timestamp'].strftime('%H:%M')} : {row[metric_column]:.2f} (Expected Range: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})"
                        )
                additional_details = "<br>".join(additional_details_parts)

                email_subject = f"Core - RG Code {rg_code}: {issue_type} - ({issue_reason})"
                
                chosen_plot_path = plot_paths_dict.get('36_Hours') if plot_paths_dict else None
                if not chosen_plot_path:
                    chosen_plot_path = plot_paths_dict.get('30_Days') if plot_paths_dict else None
                
                body_params = {
                    'node_name': node_name,
                    'kpi_name': metric_column if metric_column != 'total_rg' else 'TOTAL SG/RG',
                    'alert_time': alert_time,
                    'severity': severity,
                    'summary': summary,
                    'additional_details': additional_details,
                    'system_id': self.config.system_id,
                    'num_recent_anomalies': str(num_recent_anomalies),
                    'detection_period': f"{self.config.recent_hours} hours",
                    'ticket_number': f"TT-{datetime.now().strftime('%Y')}-{str(hash(node_name + alert_time) % 100000).zfill(6)}",
                    'plot_cid': 'node_plot_cid',
                }

                to_recipients = [test_email_recipient] if test_email_recipient else None
                cc_recipients = [] if test_email_recipient else None

                email_result = self.email_service.send_alert_email(
                    email_subject, 
                    body_params, 
                    chosen_plot_path,
                    to_override=to_recipients,
                    cc_override=cc_recipients
                )
                logger.info(f"✉️ Email status: {'✅ Sent' if email_result else '❌ Failed'}")
            else:
                logger.info(f"✅ No alert needed - {num_recent_anomalies} anomalies below threshold (> {self.config.min_anomalies_for_alert})")
            
            self._clean_output_directory()
            
            self.processed_nodes_count += 1
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❗ Error processing node {node_name}: {e}", exc_info=True)
            logger.info("=" * 80 + "\n")
            
    def _create_alarm_summary(self, node_name: str, metric_column: str, num_anomalies: int, anomalies_df: pd.DataFrame) -> None:
        """Create a special summary for alarm cases."""
        if anomalies_df.empty:
            return
            
        # Calculate some statistics about the anomalies
        try:
            avg_value = anomalies_df[metric_column].mean()
            max_value = anomalies_df[metric_column].max()
            min_value = anomalies_df[metric_column].min()
            
            # Calculate how far above/below forecast the values are (as percentage)
            anomalies_df['pct_diff'] = 100 * (anomalies_df[metric_column] - anomalies_df['yhat']) / anomalies_df['yhat']
            avg_pct_diff = anomalies_df['pct_diff'].mean()
            max_pct_diff = anomalies_df['pct_diff'].max()
            min_pct_diff = anomalies_df['pct_diff'].min()
            
            # Categorize anomalies by type (above or below forecast)
            above_forecast = anomalies_df[anomalies_df[metric_column] > anomalies_df['yhat']]
            below_forecast = anomalies_df[anomalies_df[metric_column] < anomalies_df['yhat']]
            
            anomaly_type = "mixed"
            if len(above_forecast) > 0 and len(below_forecast) == 0:
                anomaly_type = "all above forecast"
            elif len(below_forecast) > 0 and len(above_forecast) == 0:
                anomaly_type = "all below forecast"
                
            # Print special alarm summary
            logger.info(f"\n{'🚨 ALARM SUMMARY 🚨':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Node:                  {node_name}")
            logger.info(f"KPI:                   {metric_column}")
            logger.info(f"Anomaly count:         {num_anomalies}")
            logger.info(f"Anomaly type:          {anomaly_type}")
            logger.info(f"Average value:         {avg_value:.2f}")
            logger.info(f"Min/Max value:         {min_value:.2f} / {max_value:.2f}")
            logger.info(f"Avg deviation:         {avg_pct_diff:.2f}%")
            logger.info(f"Min/Max deviation:     {min_pct_diff:.2f}% / {max_pct_diff:.2f}%")
            
            # Suggestion based on anomaly pattern
            if anomaly_type == "all above forecast":
                logger.info(f"Suggestion:            Traffic significantly above normal patterns. Check for unusual demand or possible routing changes.")
            elif anomaly_type == "all below forecast":
                logger.info(f"Suggestion:            Traffic significantly below normal patterns. Possible service degradation or outage.")
            else:
                logger.info(f"Suggestion:            Irregular traffic patterns detected. Monitor for service instability.")
                
        except Exception as e:
            logger.warning(f"Could not create detailed alarm summary: {e}")
    
    def _get_anomaly_type(self, anomalies_df: pd.DataFrame, metric_column: str) -> str:
        """Determine anomaly type based on data."""
        if anomalies_df.empty:
            return "N/A"
            
        # Check if we're dealing with dl/ul specific anomalies or general anomalies
        if 'dl_yhat' in anomalies_df.columns and metric_column == 'rg_downlink':
            yhat_col = 'dl_yhat'
        elif 'ul_yhat' in anomalies_df.columns and metric_column == 'rg_uplink':
            yhat_col = 'ul_yhat'
        elif 'yhat' in anomalies_df.columns:
            yhat_col = 'yhat'
        else:
            logger.warning(f"No forecast column found in anomalies dataframe for {metric_column}")
            return "Anomalies Detected"
            
        above_forecast = anomalies_df[anomalies_df[metric_column] > anomalies_df[yhat_col]]
        below_forecast = anomalies_df[anomalies_df[metric_column] < anomalies_df[yhat_col]]
        if len(above_forecast) > 0 and len(below_forecast) == 0:
            return "All above forecast"
        elif len(below_forecast) > 0 and len(above_forecast) == 0:
            return "All below forecast"
        return "Anomalies Exceed Threshold"

    def _clean_output_directory(self) -> None:
        """Clean the output/charts directory after analysis."""
        try:
            output_dir = self.config.output_dir
            if output_dir.exists():
                files = list(output_dir.glob('*.png'))
                if files:
                    logger.info(f"Cleaning output directory: Removing {len(files)} plot files")
                    for file in files:
                        file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning output directory: {e}")

    def run_workflow(self, test_mode: bool = False, use_rg_code_analysis: bool = True) -> None:
        """Run the full anomaly detection workflow for all relevant nodes or RG codes.
        
        Args:
            test_mode: Whether to run in test mode (sending emails to test recipient)
            use_rg_code_analysis: Whether to use the new RG code-based analysis (True) or the 
                                original node-based analysis (False)
        """
        start_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"{'🚀 ANOMALY DETECTION WORKFLOW STARTED 🚀':^80}")
        logger.info("=" * 80)
        
        logger.info(f"\n{'🔧 OPERATION MODE':^80}")
        logger.info(f"{'-' * 40:^80}")
        test_mode_status = '✅ ENABLED' if test_mode else '❌ DISABLED'
        logger.info(f"{'🧪 Test mode:':<30} {test_mode_status}")
        
        analysis_mode = 'RG Code-based Analysis' if use_rg_code_analysis else 'Node-based Analysis'
        logger.info(f"{'�� Analysis mode:':<30} {analysis_mode}")
        
        if test_mode:
            test_recipient = self.config.test_recipient
            if test_recipient:
                logger.info(f"{'📧 Test recipient:':<30} {test_recipient}")
            else:
                logger.warning(f"{'⚠️ Warning:':<30} Test mode enabled but no test recipient configured")
                
        # Check prerequisites
        logger.info(f"\n{'🛠️ PREREQUISITES CHECK':^80}")
        logger.info(f"{'-' * 40:^80}")
        
        prerequisite_checks = []
        
        # Check for Oracle DB
        if not ORACLE_AVAILABLE:
            prerequisite_checks.append(("Oracle DB", "❌ MISSING", "Database access not available. Cannot fetch data."))
        else:
            prerequisite_checks.append(("Oracle DB", "✅ AVAILABLE", "Database access is ready."))
            
        # Check for Prophet
        if not PROPHET_AVAILABLE:
            prerequisite_checks.append(("Prophet", "❌ MISSING", "Cannot perform anomaly detection."))
        else:
            prerequisite_checks.append(("Prophet", "✅ AVAILABLE", "Anomaly detection engine is ready."))
            
        # Check database connection
        db_status = "❓ UNKNOWN"
        db_message = "Connection not tested yet."
        
        if ORACLE_AVAILABLE:
            db_conn_test = self.db_service.test_connection()
            if db_conn_test:
                db_status = "✅ CONNECTED"
                db_message = "Successfully connected to database."
            else:
                db_status = "❌ FAILED"
                db_message = "Could not connect to database."
                
        prerequisite_checks.append(("Database Connection", db_status, db_message))
        
        # Display prerequisites
        col_width = 25
        for prereq, status, message in prerequisite_checks:
            logger.info(f"{prereq:<{col_width}} {status:<15} {message}")
        
        # Check if we can proceed
        can_proceed = ORACLE_AVAILABLE and PROPHET_AVAILABLE
        if ORACLE_AVAILABLE:
            can_proceed = can_proceed and self.db_service.test_connection()
            
        if not can_proceed:
            logger.critical(f"\n{'❌ WORKFLOW CANNOT PROCEED':^80}")
            logger.critical(f"{'One or more critical prerequisites are not met.':^80}")
            self._log_summary(start_time)
            return
        else:
            logger.info(f"\n{'✅ ALL PREREQUISITES MET - PROCEEDING WITH WORKFLOW':^80}")

        try:
            if use_rg_code_analysis:
                # RG Code-based workflow
                logger.info(f"\n{'📋 COLLECTING RG CODES TO PROCESS':^80}")
                logger.info(f"{'-' * 40:^80}")
                
                # Get all RG codes
                rg_codes = self.db_service.get_distinct_rg_codes(hours_back=self.config.recent_hours * 2)
                if not rg_codes:
                    logger.warning(f"{'⚠️ No active RG codes found':^80}")
                    logger.warning(f"{'No RG codes were found to process in the recent time window.':^80}")
                    self._log_summary(start_time)
                    return
                
                logger.info(f"🔍 Found {len(rg_codes)} distinct RG codes for analysis")
                
                if self.config.max_nodes > 0:
                    original_count = len(rg_codes)
                    rg_codes = rg_codes[:self.config.max_nodes]
                    logger.info(f"⚙️ LIMIT APPLIED: Processing {len(rg_codes)} of {original_count} RG codes (max_nodes={self.config.max_nodes})")
                
                test_recipient = self.config.test_recipient if test_mode else None
                
                logger.info(f"\n{'🔄 BEGINNING RG CODE PROCESSING':^80}")
                logger.info(f"{'-' * 40:^80}")
                logger.info(f"Will process {len(rg_codes)} RG codes")
                
                total_rg = len(rg_codes)
                processed_rg = 0
                self._print_progress_bar(processed_rg, total_rg, prefix="RG Progress")
                
                # Process each RG code
                for i, rg_code in enumerate(rg_codes, 1):
                    logger.info(f"\n📌 Processing RG code {i}/{len(rg_codes)}: {rg_code}")
                    self.run_rg_code_analysis(rg_code, test_email_recipient=test_recipient)
                    processed_rg = i
                    self._print_progress_bar(processed_rg, total_rg, prefix="RG Progress")
                    
                    # Break if max_nodes limit reached (redundant check)
                    if self.config.max_nodes > 0 and i >= self.config.max_nodes:
                        logger.info(f"🛑 Reached max_nodes limit ({self.config.max_nodes}). Stopping further processing.")
                        break
            else:
                # Original node-based workflow
                logger.info(f"\n{'📋 COLLECTING NODES TO PROCESS':^80}")
                logger.info(f"{'-' * 40:^80}")
                
                # Get all RG nodes for the recent time window
                rg_nodes_to_process = self.db_service.get_distinct_rg_nodes(hours_back=self.config.recent_hours * 2)
                if not rg_nodes_to_process:
                    logger.warning(f"{'⚠️ No active RG nodes found':^80}")
                    logger.warning(f"{'No RG nodes were found to process in the recent time window.':^80}")
                    self._log_summary(start_time)
                    return

                logger.info(f"🔍 Found {len(rg_nodes_to_process)} distinct RG nodes for potential analysis")

                if self.config.max_nodes > 0:
                    original_count = len(rg_nodes_to_process)
                    rg_nodes_to_process = rg_nodes_to_process[:self.config.max_nodes]
                    logger.info(f"⚙️ LIMIT APPLIED: Processing {len(rg_nodes_to_process)} of {original_count} RG nodes (max_nodes={self.config.max_nodes})")
                
                test_recipient = self.config.test_recipient if test_mode else None

                logger.info(f"\n{'🔄 BEGINNING NODE PROCESSING':^80}")
                logger.info(f"{'-' * 40:^80}")
                logger.info(f"Will process {len(rg_nodes_to_process)} RG nodes with KPI: 'total_rg'")

                # Group RG nodes by their base names for aggregated analysis
                rg_groups = {}
                for rg_node in rg_nodes_to_process:
                    rg_base = self.db_service.get_rg_base_name(rg_node)
                    if rg_base not in rg_groups:
                        rg_groups[rg_base] = []
                    rg_groups[rg_base].append(rg_node)
                
                logger.info(f"🔗 Grouped {len(rg_nodes_to_process)} RG nodes into {len(rg_groups)} RG groups:")
                for rg_base, nodes in rg_groups.items():
                    logger.info(f"   - {rg_base}: {len(nodes)} nodes")

                processed_rg_groups = 0
                for i, (rg_base, rg_nodes_in_group) in enumerate(rg_groups.items(), 1):
                    logger.info(f"\n📌 Processing RG group {i}/{len(rg_groups)}: {rg_base}")
                    logger.info(f"   - RG nodes in group: {len(rg_nodes_in_group)}")
                    
                    # Get all CG nodes associated with this RG base
                    cg_nodes = self.db_service.get_associated_cg_nodes(rg_base, hours_back=self.config.recent_hours * 2)
                    logger.info(f"   - Associated CG nodes: {len(cg_nodes)}")
                    
                    # Run aggregated analysis for this RG group
                    self.run_rg_aggregated_analysis(
                        rg_base=rg_base,
                        rg_nodes=rg_nodes_in_group,
                        cg_nodes=cg_nodes,
                        metric_column='total_rg',
                        test_email_recipient=test_recipient
                    )
                    
                    processed_rg_groups += 1
                    if self.config.max_nodes > 0 and processed_rg_groups >= self.config.max_nodes:
                        logger.info(f"🛑 Reached max_nodes limit ({self.config.max_nodes}). Stopping further processing.")
                        break
        
        except Exception as e:
            logger.error(f"\n{'❗ CRITICAL ERROR':^80}")
            logger.error(f"{str(e):^80}")
            logger.error("See traceback for details:", exc_info=True)
        
        finally:
            self._log_summary(start_time)

    def _print_progress_bar(self, current: int, total: int, bar_len: int = 40, prefix: str = "Progress") -> None:
        """Render a single-line progress bar at the bottom of the terminal.

        Args:
            current: Number of items processed so far (1-based or 0-based is fine)
            total: Total number of items to process
            bar_len: Length of the bar in characters
            prefix: Text prefix for the bar
        """
        try:
            current = max(0, min(current, total))
            ratio = (current / total) if total > 0 else 1.0
            filled_len = int(round(bar_len * ratio))
            bar = '#' * filled_len + '-' * (bar_len - filled_len)
            percent = int(round(ratio * 100))
            sys.stdout.write(f"\r{prefix}: |{bar}| {percent:3d}% ({current}/{total})")
            sys.stdout.flush()
            if current >= total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        except Exception:
            # Never let progress bar break the workflow
            pass

    def _apply_increment_decrement_policy(self, rg_code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply anomaly policy: care only decrements, except for configured RGs where increments are
        considered if above percentage threshold and sustained for configured days.

        Expects df to contain columns:
          timestamp, rg_downlink, rg_uplink,
          dl_anomaly, dl_yhat, dl_yhat_lower, dl_yhat_upper,
          ul_anomaly, ul_yhat, ul_yhat_lower, ul_yhat_upper
        """
        if df.empty:
            return df

        result = df.copy()
        # Define negative anomaly masks (below forecast)
        dl_neg = (result.get('dl_anomaly', False)) & (result.get('rg_downlink', 0) < result.get('dl_yhat', np.nan))
        ul_neg = (result.get('ul_anomaly', False)) & (result.get('rg_uplink', 0) < result.get('ul_yhat', np.nan))

        # Default: only decrements are cared
        new_dl_anom = dl_neg.copy()
        new_ul_anom = ul_neg.copy()

        # Exception RGs: consider increments if above threshold and sustained
        if str(rg_code) in self.config.increment_exception_rg:
            eps = 1e-9
            # Positive anomaly masks (above forecast)
            dl_pos = (result.get('dl_anomaly', False)) & (result.get('rg_downlink', 0) > result.get('dl_yhat', np.nan))
            ul_pos = (result.get('ul_anomaly', False)) & (result.get('rg_uplink', 0) > result.get('ul_yhat', np.nan))

            # Percent increments relative to forecast (yhat)
            dl_inc_pct = 100.0 * (result.get('rg_downlink', 0) - result.get('dl_yhat', np.nan)) / (np.maximum(result.get('dl_yhat', np.nan), eps))
            ul_inc_pct = 100.0 * (result.get('rg_uplink', 0) - result.get('ul_yhat', np.nan)) / (np.maximum(result.get('ul_yhat', np.nan), eps))

            dl_pos_above = dl_pos & (dl_inc_pct >= self.config.increment_percentage)
            ul_pos_above = ul_pos & (ul_inc_pct >= self.config.increment_percentage)

            # Check sustained condition over the last N days
            end_time = pd.to_datetime(result['timestamp'].max())
            start_time = end_time - timedelta(days=self.config.increment_continue_days)
            in_window = pd.to_datetime(result['timestamp']) >= start_time

            # Require all hours in window to be positive increments above threshold for 'steady state'
            window_idx = result.index[in_window]
            sustained_dl = False
            sustained_ul = False
            if len(window_idx) > 0:
                sustained_dl = bool((dl_pos_above[in_window]).all())
                sustained_ul = bool((ul_pos_above[in_window]).all())

            # If either DL or UL satisfied sustained increment condition, include those increments
            if sustained_dl:
                new_dl_anom = new_dl_anom | dl_pos_above
            if sustained_ul:
                new_ul_anom = new_ul_anom | ul_pos_above

        # Apply back
        result['dl_anomaly'] = new_dl_anom.fillna(False)
        result['ul_anomaly'] = new_ul_anom.fillna(False)
        result['anomaly'] = result['dl_anomaly'] | result['ul_anomaly']
        return result


def main():
    """Main function to run the anomaly detection workflow."""
    
    # Check for command-line arguments
    # Example: python final_anomaly_workflow.py --test
    #          python final_anomaly_workflow.py --node SG_CORE_TH1_GE_0_0_1 --test
    #          python final_anomaly_workflow.py --max-nodes 5
    #          python final_anomaly_workflow.py --rg-code 70 --test
    #          python final_anomaly_workflow.py --all-rg-codes --test

    import argparse
    parser = argparse.ArgumentParser(description="Network KPI Anomaly Detection Workflow.")
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run in test mode. Emails will be sent to EMAIL_TEST_RECIPIENT if configured."
    )
    parser.add_argument(
        "--node", 
        type=str, 
        help="Run analysis for a specific node name."
    )
    parser.add_argument(
        "--kpi", 
        type=str, 
        default="total_rg", 
        help="Specify the KPI column to analyze (default: total_rg)."
    )
    # Allow overriding max_nodes from CLI
    config_temp = ConfigManager() # To get default max_nodes
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=config_temp.max_nodes, # Default from ConfigManager
        help=f"Maximum number of nodes to process (0 for all, default from env: {config_temp.max_nodes}). Overrides .env value if specified."
    )
    # Add new argument for RG code analysis
    parser.add_argument(
        "--rg-code", 
        type=str, 
        help="Run analysis for a specific RG code (e.g., '70')."
    )
    # Add argument to run all RG codes
    parser.add_argument(
        "--all-rg-codes", 
        action="store_true", 
        help="Run analysis for all RG codes in the system."
    )
    # Add argument to select analysis mode
    parser.add_argument(
        "--use-original-mode",
        action="store_true",
        help="Use the original node-based analysis instead of the new RG code-based analysis."
    )
    
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info(f"{'🛠️ WORKFLOW SETTINGS 🛠️':^80}")
    logger.info("=" * 80)
    
    logger.info(f"\n{'📋 COMMAND-LINE ARGUMENTS':^80}")
    logger.info(f"{'-' * 40:^80}")
    
    # Format command-line args nicely
    cli_args = [
        ('--test', '✅ Active' if args.test else '❌ Inactive', 'Run in test mode'),
        ('--node', args.node if args.node else 'Not specified', 'Single node analysis'),
        ('--rg-code', args.rg_code if args.rg_code else 'Not specified', 'RG code analysis'),
        ('--all-rg-codes', '✅ Active' if args.all_rg_codes else '❌ Inactive', 'Process all RG codes'),
        ('--use-original-mode', '✅ Active' if args.use_original_mode else '❌ Inactive', 'Use original node-based analysis'),
        ('--kpi', args.kpi, 'KPI to analyze'),
        ('--max-nodes', f"{args.max_nodes} {'(default)' if args.max_nodes == config_temp.max_nodes else '(custom)'}", 
         '0 = process all nodes, >0 = limit to specified count')
    ]
    
    for arg, value, description in cli_args:
        logger.info(f"{arg:<20} {value:<40} {description}")
    
    workflow = AnomalyWorkflow()
    
    # Update config from CLI args if provided
    if args.max_nodes != config_temp.max_nodes:
        logger.info(f"\n{'⚙️ CONFIGURATION OVERRIDE':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Overriding 'max_nodes' from CLI: {args.max_nodes} (was {workflow.config.max_nodes})")
        workflow.config.max_nodes = args.max_nodes

    # Process a single RG code
    if args.rg_code:
        logger.info(f"\n{'🎯 RUNNING RG CODE ANALYSIS':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"RG Code: {args.rg_code}")
        if args.test:
            logger.info(f"Mode: Test mode (email to {workflow.config.test_recipient or 'default recipients'})")
        
        test_recipient = workflow.config.test_recipient if args.test else None
        if args.test and not test_recipient:
             logger.warning(f"{'⚠️ WARNING: Test mode enabled but no test recipient configured':^80}")
        
        # For single RG code, we call _log_summary manually after analysis
        single_start_time = datetime.now()
        workflow.run_rg_code_analysis(args.rg_code, test_email_recipient=test_recipient)
        workflow._log_summary(single_start_time)
    
    # Process all RG codes
    elif args.all_rg_codes:
        logger.info(f"\n{'🌐 RUNNING ALL RG CODES ANALYSIS':^80}")
        logger.info(f"{'-' * 40:^80}")
        if args.test:
            logger.info(f"Mode: Test mode (email to {workflow.config.test_recipient or 'default recipients'})")
        
        test_recipient = workflow.config.test_recipient if args.test else None
        if args.test and not test_recipient:
             logger.warning(f"{'⚠️ WARNING: Test mode enabled but no test recipient configured':^80}")
             
        # Get all RG codes
        rg_codes = workflow.db_service.get_distinct_rg_codes(hours_back=workflow.config.recent_hours * 2)
        if not rg_codes:
            logger.warning(f"No RG codes found in the database for the last {workflow.config.recent_hours * 2} hours. Exiting.")
            return
            
        logger.info(f"Found {len(rg_codes)} distinct RG codes to analyze")
        logger.info(f"Sample RG codes: {', '.join(rg_codes[:5])}{'...' if len(rg_codes) > 5 else ''}")
        
        # Start time for summary
        all_codes_start_time = datetime.now()
        
        # Apply max_nodes limit if specified
        if workflow.config.max_nodes > 0 and workflow.config.max_nodes < len(rg_codes):
            original_count = len(rg_codes)
            rg_codes = rg_codes[:workflow.config.max_nodes]
            logger.info(f"⚙️ LIMIT APPLIED: Processing {len(rg_codes)} of {original_count} RG codes (max_nodes={workflow.config.max_nodes})")
        
        # Process each RG code
        total_rg = len(rg_codes)
        processed_rg = 0
        workflow._print_progress_bar(processed_rg, total_rg, prefix="RG Progress")
        for i, rg_code in enumerate(rg_codes, 1):
            logger.info(f"\n📌 Processing RG code {i}/{len(rg_codes)}: {rg_code}")
            workflow.run_rg_code_analysis(rg_code, test_email_recipient=test_recipient)
            processed_rg = i
            workflow._print_progress_bar(processed_rg, total_rg, prefix="RG Progress")
        
        # Log summary
        workflow._log_summary(all_codes_start_time)
        
    # Process a specific node
    elif args.node:
        logger.info(f"\n{'🎯 RUNNING SINGLE NODE/RG GROUP ANALYSIS':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Node: {args.node}")
        logger.info(f"KPI:  {args.kpi}")
        if args.test:
            logger.info(f"Mode: Test mode (email to {workflow.config.test_recipient or 'default recipients'})")
        
        test_recipient_single_node = workflow.config.test_recipient if args.test else None
        if args.test and not test_recipient_single_node:
             logger.warning(f"{'⚠️ WARNING: Test mode enabled but no test recipient configured':^80}")
        
        # For single node, we call _log_summary manually after analysis
        single_node_start_time = datetime.now()
        
        # Check if this is an RG node for group analysis
        if args.node.startswith('TH1VCGH1'):
            logger.info(f"{'Detected RG node - will run RG group analysis':^80}")
            rg_base = workflow.db_service.get_rg_base_name(args.node)
            
            # Get all RG nodes with the same base
            all_rg_nodes = workflow.db_service.get_distinct_rg_nodes(hours_back=workflow.config.recent_hours * 2)
            rg_nodes_in_group = [node for node in all_rg_nodes if workflow.db_service.get_rg_base_name(node) == rg_base]
            
            # Get associated CG nodes
            cg_nodes = workflow.db_service.get_associated_cg_nodes(rg_base, hours_back=workflow.config.recent_hours * 2)
            
            logger.info(f"RG group '{rg_base}': {len(rg_nodes_in_group)} RG nodes + {len(cg_nodes)} CG nodes")
            
            workflow.run_rg_aggregated_analysis(
                rg_base=rg_base,
                rg_nodes=rg_nodes_in_group,
                cg_nodes=cg_nodes,
                metric_column=args.kpi,
                test_email_recipient=test_recipient_single_node
            )
        else:
            logger.info(f"{'Running individual node analysis':^80}")
            workflow.run_node_analysis(args.node, metric_column=args.kpi, test_email_recipient=test_recipient_single_node)
        
        workflow._log_summary(single_node_start_time) 
    else:
        logger.info(f"\n{'🌐 RUNNING FULL WORKFLOW':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Mode: {'Test mode' if args.test else 'Production mode'}")
        logger.info(f"Analysis will process all available {'nodes' if args.use_original_mode else 'RG codes'} (limited to {workflow.config.max_nodes if workflow.config.max_nodes > 0 else 'ALL'})")
        
        # Run the full workflow with the selected analysis mode
        use_rg_code_analysis = not args.use_original_mode  # Default to new RG code analysis unless --use-original-mode is specified
        workflow.run_workflow(test_mode=args.test, use_rg_code_analysis=use_rg_code_analysis)

if __name__ == "__main__":
    main() 