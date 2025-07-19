#!/usr/bin/env python3
"""
Final Production-Ready Anomaly Detection Workflow

This script implements the complete network KPI anomaly detection workflow:
1. Loads environment variables from .env file
2. Uses email template from templates/email_templates/sg_rg_traffic_alert.html
3. Includes improved x-axis plotting for better readability
4. Email-compatible template rendering
5. Comprehensive error handling and logging

Usage:
    python scripts/final_anomaly_workflow.py

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
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
    # Load .env file
    env_path = Path(__file__).parent.parent / '.env'
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
        self.db_password = os.getenv('DATABASE_PASSWORD', 'tools_ml')
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
        self.min_anomalies_for_alert = int(os.getenv('MIN_ANOMALIES_FOR_ALERT', '4')) # New: min anomalies for alert
        self.historical_days = int(os.getenv('HISTORICAL_DAYS', '30'))
        
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
        """Create a default email template if none exists."""
        default_template_path = self.email_template_dir / "sg_rg_traffic_alert.html"
        
        if not default_template_path.exists():
            logger.info(f"Creating default email template at {default_template_path}")
            
            default_template = """
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
            
            try:
                with open(default_template_path, 'w', encoding='utf-8') as f:
                    f.write(default_template)
                logger.info(f"✅ Default email template created successfully")
            except Exception as e:
                logger.error(f"Failed to create default email template: {e}")

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


class VisualizationService:
    """Service for creating and saving visualizations."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize visualization service."""
        self.output_dir = config.output_dir
        # plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style # Commented out to use default

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
    ) -> Optional[str]:
        """Plot KPI data with anomalies and forecast if available."""
        if df.empty:
            logger.warning(f"No data to plot for node {node_name}, metric {metric_column}.")
            return None

        plt.figure(figsize=(15, 7))
        ax = plt.gca()

        # Plot actual data
        sns.lineplot(x='timestamp', y=metric_column, data=df, label='Actual KPI', ax=ax, color='blue', linewidth=1.5)

        # Plot forecast if available
        if forecast_df is not None and not forecast_df.empty:
            sns.lineplot(x='ds', y='yhat', data=forecast_df, label='Forecast (yhat)', ax=ax, color='orange', linestyle='--', linewidth=1.5)
            ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')

        # Highlight anomalies
        anomalies = df[df['anomaly'] == True]
        if not anomalies.empty:
            sns.scatterplot(x='timestamp', y=metric_column, data=anomalies, color='red', s=100, label='Anomaly', ax=ax, marker='o', zorder=5)
        
        df_len = len(df['timestamp']) if 'timestamp' in df else 0
        self._format_plot(ax, f"KPI Anomaly Detection for {node_name} - {metric_column}", df_len)
        
        # Save plot
        plot_filename = f"{node_name}_{metric_column}_anomaly_plot.png"
        plot_path = self.output_dir / plot_filename
        try:
            plt.savefig(plot_path, dpi=100, bbox_inches='tight') # Lower DPI for email size
            plt.close()
            logger.info(f"Plot saved to {plot_path}")
            return str(plot_path)
        except Exception as e:
            logger.error(f"Failed to save plot {plot_path}: {e}")
            plt.close()
            return None

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
        self.email_template_path = config.email_template_dir / "sg_rg_traffic_alert.html"

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

    def send_alert_email(self, subject: str, body_params: Dict[str, str], plot_path: Optional[str], to_override: Optional[List[str]] = None, cc_override: Optional[List[str]] = None) -> bool:
        """Send an email alert with an embedded plot."""
        
        html_template = self._load_email_template() # This is the original template loading
        if html_template is None:
             logger.error("Cannot send email: HTML template failed to load.")
             # Fallback to the purely code-generated email if template is missing
             if 'plot_cid' not in body_params and plot_path: # Ensure plot_cid is ready for _create_email_compatible_body
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
        else: # Use the sg_rg_traffic_alert.html template
            # Ensure plot_cid is set for template substitution
            if 'plot_cid' not in body_params and plot_path:
                body_params['plot_cid'] = Path(plot_path).name if plot_path else "plot_image"
            
            # Ensure all expected keys are present, provide defaults if not
            default_params = {
                'node_name': 'N/A', 'kpi_name': 'N/A', 'alert_time': 'N/A',
                'severity': 'INFO', 'summary': 'No summary.', 'additional_details': 'N/A',
                'system_id': self.config.system_id,
                'severity_color': '#1976D2', # Default blue for INFO
                'header_color': '#0D47A1' # Default dark blue for INFO
            }
            # Map severity to colors for the template
            severity_color_map = { "CRITICAL": "#D32F2F", "WARNING": "#FFA000", "INFO": "#1976D2" }
            header_color_map = { "CRITICAL": "#B71C1C", "WARNING": "#F57C00", "INFO": "#0D47A1" }
            
            current_severity = body_params.get('severity', 'INFO').upper()
            body_params['severity_color'] = severity_color_map.get(current_severity, severity_color_map['INFO'])
            body_params['header_color'] = header_color_map.get(current_severity, header_color_map['INFO'])

            final_params = {**default_params, **body_params} # Merge defaults with provided params
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

        if plot_path and Path(plot_path).exists():
            try:
                with open(plot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', f"<{body_params['plot_cid']}>") # Use the same CID as in HTML
                    img.add_header('Content-Disposition', 'inline', filename=Path(plot_path).name)
                    msg.attach(img)
            except Exception as e:
                logger.error(f"Failed to attach plot image {plot_path}: {e}")
        else:
            logger.warning(f"Plot path {plot_path} not found or not provided. Sending email without image.")

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
                # Try standard SMTP on port 25 or 587
                fallback_port = 587  # Common non-SSL SMTP port for STARTTLS
                logger.info(f"Attempting to send email without SSL (fallback to port {fallback_port})...")
                
                with smtplib.SMTP(self.config.smtp_host, fallback_port) as server:
                    server.ehlo()
                    # Try STARTTLS if available
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
            
            # Summarized results - avoid repetitive info
            logger.info(f"\n{'ANALYSIS RESULTS':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Time window: Last {self.config.recent_hours} hours | Alert threshold: > {self.config.min_anomalies_for_alert}")
            logger.info(f"Anomalies detected: {num_recent_anomalies}")
            
            # Determine if alert is needed
            alert_needed = num_recent_anomalies > self.config.min_anomalies_for_alert
            
            # 3. Only generate visualization if alert is needed
            plot_path = None
            if alert_needed:
                if num_recent_anomalies > 0:
                    logger.info(f"\n{'ANOMALY DETAILS':^80}")
                    logger.info(f"{'-' * 40:^80}")
                    logger.info(f"First anomaly: {recent_anomalies['timestamp'].min().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    logger.info(f"Last anomaly:  {recent_anomalies['timestamp'].max().strftime('%Y-%m-%d %H:%M') if not recent_anomalies.empty else 'N/A'}")
                    
                    # Create special summary for alarm cases
                    self._create_alarm_summary(node_name, metric_column, num_recent_anomalies, recent_anomalies)
                
                # Generate plot only when alert is needed
                logger.info(f"\n📈 Generating visualization for alert...")
                plot_path = self.viz_service.plot_kpi_with_anomalies(df_with_anomalies, node_name, metric_column, forecast_df)
                if plot_path:
                    logger.info(f"✅ Plot saved to: {plot_path}")
            
            # 4. Send email alert if significant anomalies found
            logger.info(f"\n{'ALERT STATUS':^80}")
            logger.info(f"{'-' * 40:^80}")
            
            # Condition: strictly more than min_anomalies_for_alert
            if alert_needed:
                severity = "CRITICAL" if num_recent_anomalies >= self.config.min_anomalies_for_alert * 2 else "WARNING"
                severity_icon = "🔴" if severity == "CRITICAL" else "🟠"
                
                logger.info(f"{severity_icon} {severity} ALERT will be sent")
                logger.info(f"📧 Recipients: {test_email_recipient if test_email_recipient else self.config.email_to}")
                
                alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create a more detailed summary for the email
                summary = f"{num_recent_anomalies} anomalies detected for KPI '{metric_column}' on node '{node_name}' in the last {self.config.recent_hours} hours."
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
                    for _, row in recent_anomalies.head(5).iterrows(): # Show up to 5 recent anomalies
                         additional_details_parts.append(
                             f"- {row['timestamp'].strftime('%H:%M')} : {row[metric_column]:.2f} (Expected Range: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})"
                         )
                
                additional_details = "<br>".join(additional_details_parts)

                email_subject = f"{severity} ALERT: Anomalies Detected on {node_name} for {metric_column}"
                
                body_params = {
                    'node_name': node_name,
                    'kpi_name': metric_column,
                    'alert_time': alert_time,
                    'severity': severity,
                    'summary': summary,
                    'additional_details': additional_details,
                    'plot_cid': Path(plot_path).name if plot_path else "plot_image",
                    'system_id': self.config.system_id,
                    'num_recent_anomalies': str(num_recent_anomalies),
                    'detection_period': f"{self.config.recent_hours} hours",
                    # New ticket parameters
                    'ticket_number': f"TT-{datetime.now().strftime('%Y')}-{str(hash(node_name + alert_time) % 100000).zfill(6)}",
                    'ticket_url': f"https://10.201.6.13/KM_UCMS_TT/ticket/TT-{datetime.now().strftime('%Y')}-{str(hash(node_name + alert_time) % 100000).zfill(6)}"
                }

                to_recipients = [test_email_recipient] if test_email_recipient else None
                cc_recipients = [] if test_email_recipient else None # No CC if sending test email

                email_result = self.email_service.send_alert_email(
                    email_subject, 
                    body_params, 
                    plot_path,
                    to_override=to_recipients,
                    cc_override=cc_recipients
                )
                logger.info(f"✉️ Email status: {'✅ Sent' if email_result else '❌ Failed'}")
            else:
                logger.info(f"✅ No alert needed - {num_recent_anomalies} anomalies below threshold (> {self.config.min_anomalies_for_alert})")
            
            # Clean output/charts directory after analyzing each case
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

    def run_workflow(self, test_mode: bool = False) -> None:
        """Run the full anomaly detection workflow for all relevant nodes."""
        start_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"{'🚀 ANOMALY DETECTION WORKFLOW STARTED 🚀':^80}")
        logger.info("=" * 80)
        
        logger.info(f"\n{'🔧 OPERATION MODE':^80}")
        logger.info(f"{'-' * 40:^80}")
        test_mode_status = '✅ ENABLED' if test_mode else '❌ DISABLED'
        logger.info(f"{'🧪 Test mode:':<30} {test_mode_status}")
        
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
            logger.info(f"\n{'📋 COLLECTING NODES TO PROCESS':^80}")
            logger.info(f"{'-' * 40:^80}")
            
            nodes_to_process = self.db_service.get_distinct_nodes(hours_back=self.config.recent_hours * 2)
            if not nodes_to_process:
                logger.warning(f"{'⚠️ No active nodes found':^80}")
                logger.warning(f"{'No nodes were found to process in the recent time window.':^80}")
                self._log_summary(start_time)
                return

            logger.info(f"🔍 Found {len(nodes_to_process)} distinct nodes for potential analysis")

            if self.config.max_nodes > 0:
                original_count = len(nodes_to_process)
                nodes_to_process = nodes_to_process[:self.config.max_nodes]
                logger.info(f"⚙️ LIMIT APPLIED: Processing {len(nodes_to_process)} of {original_count} nodes (max_nodes={self.config.max_nodes})")
            
            test_recipient = self.config.test_recipient if test_mode else None

            logger.info(f"\n{'🔄 BEGINNING NODE PROCESSING':^80}")
            logger.info(f"{'-' * 40:^80}")
            logger.info(f"Will process {len(nodes_to_process)} nodes with KPI: 'total_rg'")

            for i, node_name in enumerate(nodes_to_process, 1):
                logger.info(f"\n📌 Processing node {i}/{len(nodes_to_process)}: {node_name}")
                self.run_node_analysis(node_name, metric_column='total_rg', test_email_recipient=test_recipient)
                if self.config.max_nodes > 0 and self.processed_nodes_count >= self.config.max_nodes:
                    logger.info(f"🛑 Reached max_nodes limit ({self.config.max_nodes}). Stopping further processing.")
                    break
        
        except Exception as e:
            logger.error(f"\n{'❗ CRITICAL ERROR':^80}")
            logger.error(f"{str(e):^80}")
            logger.error("See traceback for details:", exc_info=True)
        
        finally:
            self._log_summary(start_time)


def main():
    """Main function to run the anomaly detection workflow."""
    
    # Check for command-line arguments
    # Example: python final_anomaly_workflow.py --test
    #          python final_anomaly_workflow.py --node SG_CORE_TH1_GE_0_0_1 --test
    #          python final_anomaly_workflow.py --max-nodes 5

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
    
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info(f"{'🛠️ WORKFLOW SETTINGS 🛠️':^80}")
    logger.info("=" * 80)
    
    logger.info(f"\n{'📋 COMMAND-LINE ARGUMENTS':^80}")
    logger.info(f"{'-' * 40:^80}")
    
    # Format command-line args nicely
    cli_args = [
        ('--test', '✅ Active' if args.test else '❌ Inactive', 'Run in test mode'),
        ('--node', args.node if args.node else 'Not specified (analyzing all nodes)', 'Single node analysis'),
        ('--kpi', args.kpi, 'KPI to analyze'),
        ('--max-nodes', f"{args.max_nodes} {'(default)' if args.max_nodes == config_temp.max_nodes else '(custom)'}", 
         '0 = process all nodes, >0 = limit to specified count')
    ]
    
    for arg, value, description in cli_args:
        logger.info(f"{arg:<15} {value:<40} {description}")
    
    workflow = AnomalyWorkflow()
    
    # Update config from CLI args if provided
    if args.max_nodes != config_temp.max_nodes:
        logger.info(f"\n{'⚙️ CONFIGURATION OVERRIDE':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Overriding 'max_nodes' from CLI: {args.max_nodes} (was {workflow.config.max_nodes})")
        workflow.config.max_nodes = args.max_nodes

    if args.node:
        logger.info(f"\n{'🎯 RUNNING SINGLE NODE ANALYSIS':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Node: {args.node}")
        logger.info(f"KPI:  {args.kpi}")
        if args.test:
            logger.info(f"Mode: Test mode (email to {workflow.config.test_recipient or 'default recipients'})")
        
        test_recipient_single_node = workflow.config.test_recipient if args.test else None
        if args.test and not test_recipient_single_node:
             logger.warning(f"{'⚠️ WARNING: Test mode enabled but no test recipient configured':^80}")
        
        # For single node, we call _log_summary manually after run_node_analysis
        single_node_start_time = datetime.now()
        workflow.run_node_analysis(args.node, metric_column=args.kpi, test_email_recipient=test_recipient_single_node)
        workflow._log_summary(single_node_start_time) 
    else:
        logger.info(f"\n{'🌐 RUNNING FULL WORKFLOW':^80}")
        logger.info(f"{'-' * 40:^80}")
        logger.info(f"Mode: {'Test mode' if args.test else 'Production mode'}")
        logger.info(f"Analysis will process all available nodes (limited to {workflow.config.max_nodes if workflow.config.max_nodes > 0 else 'ALL'})")
        workflow.run_workflow(test_mode=args.test)

if __name__ == "__main__":
    main() 