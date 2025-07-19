"""
Simplified configuration management for initial testing.
This version works without external dependencies.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    username: str
    password: str
    dsn: str
    port: int
    host: str
    encoding: str = "UTF-8"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


@dataclass
class TicketingConfig:
    """Ticketing system configuration."""
    api_url: str
    username: str
    password: str
    bearer_token: str
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class EmailConfig:
    """Email system configuration."""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    use_tls: bool = True
    timeout: int = 30
    recipients: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class KPIConfig:
    """Individual KPI monitoring configuration."""
    name: str
    enabled: bool = True
    detection_algorithm: str = "prophet"
    threshold_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring system configuration."""
    kpis: List[KPIConfig] = field(default_factory=list)
    data_collection: Dict[str, int] = field(default_factory=dict)
    schedule: Dict[str, str] = field(default_factory=dict)


@dataclass
class DetectionConfig:
    """Anomaly detection configuration."""
    prophet: Dict[str, Any] = field(default_factory=dict)
    anomaly_detection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileConfig:
    """File management configuration."""
    chart_output_dir: str = "./output/charts"
    log_output_dir: str = "./logs"
    template_dir: str = "./templates"
    temp_dir: str = "./temp"
    retention: Dict[str, int] = field(default_factory=dict)


class Config:
    """
    Simplified configuration manager using environment variables.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self._config_data: Dict[str, Any] = self._create_default_config()
        
        # Load config file if exists
        if config_path and Path(config_path).exists():
            try:
                self.load_config()
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
                print("Using environment variables and defaults")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration from environment variables."""
        return {
            'database': {
                'username': os.getenv('DB_USERNAME', 'tools_ml'),
                'password': os.getenv('DB_PASSWORD', ''),
                'dsn': os.getenv('DB_DSN', 'fcsouth.rightel.ir'),
                'port': int(os.getenv('DB_PORT', '1521')),
                'host': os.getenv('DB_HOST', '10.200.6.227'),
                'encoding': os.getenv('DB_ENCODING', 'UTF-8'),
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30
            },
            'ticketing': {
                'api_url': os.getenv('TICKETING_API_URL', ''),
                'username': os.getenv('TICKETING_USERNAME', ''),
                'password': os.getenv('TICKETING_PASSWORD', ''),
                'bearer_token': os.getenv('TICKETING_BEARER_TOKEN', ''),
                'timeout': 30,
                'retry_attempts': 3,
                'retry_delay': 5
            },
            'email': {
                'smtp_server': os.getenv('EMAIL_SMTP_SERVER', ''),
                'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '465')),
                'username': os.getenv('EMAIL_USERNAME', ''),
                'password': os.getenv('EMAIL_PASSWORD', ''),
                'from_address': os.getenv('EMAIL_FROM_ADDRESS', ''),
                'use_tls': os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true',
                'timeout': 30,
                'recipients': {
                    'to': [os.getenv('EMAIL_TO', '')],
                    'cc': os.getenv('EMAIL_CC', '').split(',') if os.getenv('EMAIL_CC') else []
                }
            },
            'monitoring': {
                'kpis': [
                    {
                        'name': '3G Traffic Channel Assignment Success Rate',
                        'enabled': True,
                        'detection_algorithm': 'prophet',
                        'threshold_config': {
                            'confidence_interval': 0.99,
                            'slope_threshold': 2.0,
                            'min_data_points': 24
                        }
                    },
                    {
                        'name': 'Paging Success Rate',
                        'enabled': True,
                        'detection_algorithm': 'prophet',
                        'threshold_config': {
                            'confidence_interval': 0.99,
                            'slope_threshold': 2.0,
                            'min_data_points': 24
                        }
                    }
                ],
                'data_collection': {
                    'default_range_hours': 168,
                    'minimum_range_hours': 24
                }
            },
            'detection': {
                'prophet': {
                    'interval_width': 0.99,
                    'weekly_seasonality': True,
                    'daily_seasonality': True
                },
                'anomaly_detection': {
                    'sensitivity': 0.01
                }
            },
            'files': {
                'chart_output_dir': './output/charts',
                'log_output_dir': './logs',
                'template_dir': './templates',
                'temp_dir': './temp'
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path:
            return
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Simple environment variable substitution
            content = self._substitute_env_vars(content)
            
            # Parse YAML and merge with defaults
            yaml_config = yaml.safe_load(content)
            self._merge_config(self._config_data, yaml_config)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Simple environment variable substitution."""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replace_env_var, content)
    
    def _merge_config(self, base: dict, update: dict) -> None:
        """Merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config_data.get('database', {})
        return DatabaseConfig(
            username=db_config.get('username', ''),
            password=db_config.get('password', ''),
            dsn=db_config.get('dsn', ''),
            port=int(db_config.get('port', 1521)),
            host=db_config.get('host', ''),
            encoding=db_config.get('encoding', 'UTF-8'),
            pool_size=int(db_config.get('pool_size', 5)),
            max_overflow=int(db_config.get('max_overflow', 10)),
            pool_timeout=int(db_config.get('pool_timeout', 30))
        )
    
    @property
    def ticketing(self) -> TicketingConfig:
        """Get ticketing system configuration."""
        ticket_config = self._config_data.get('ticketing', {})
        return TicketingConfig(
            api_url=ticket_config.get('api_url', ''),
            username=ticket_config.get('username', ''),
            password=ticket_config.get('password', ''),
            bearer_token=ticket_config.get('bearer_token', ''),
            timeout=int(ticket_config.get('timeout', 30)),
            retry_attempts=int(ticket_config.get('retry_attempts', 3)),
            retry_delay=int(ticket_config.get('retry_delay', 5))
        )
    
    @property
    def email(self) -> EmailConfig:
        """Get email configuration."""
        email_config = self._config_data.get('email', {})
        return EmailConfig(
            smtp_server=email_config.get('smtp_server', ''),
            smtp_port=int(email_config.get('smtp_port', 465)),
            username=email_config.get('username', ''),
            password=email_config.get('password', ''),
            from_address=email_config.get('from_address', ''),
            use_tls=bool(email_config.get('use_tls', True)),
            timeout=int(email_config.get('timeout', 30)),
            recipients=email_config.get('recipients', {})
        )
    
    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_config = self._config_data.get('monitoring', {})
        
        # Parse KPI configurations
        kpis = []
        for kpi_data in monitoring_config.get('kpis', []):
            kpis.append(KPIConfig(
                name=kpi_data.get('name', ''),
                enabled=bool(kpi_data.get('enabled', True)),
                detection_algorithm=kpi_data.get('detection_algorithm', 'prophet'),
                threshold_config=kpi_data.get('threshold_config', {})
            ))
        
        return MonitoringConfig(
            kpis=kpis,
            data_collection=monitoring_config.get('data_collection', {}),
            schedule=monitoring_config.get('schedule', {})
        )
    
    def get_enabled_kpis(self) -> List[KPIConfig]:
        """Get list of enabled KPIs for monitoring."""
        return [kpi for kpi in self.monitoring.kpis if kpi.enabled]
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            './output/charts',
            './logs',
            './temp'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config 