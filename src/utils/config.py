"""
Configuration management for the network monitoring system.

This module provides centralized configuration management with support for:
- Environment variables
- YAML configuration files
- Validation and type checking
- Secure credential handling
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    Centralized configuration manager for the network monitoring system.
    
    This class handles loading configuration from YAML files and environment
    variables, with proper validation and type checking.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self._config_data: Dict[str, Any] = {}
        self.load_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "config/config.yaml",
            "config/config.yml",
            "../config/config.yaml",
            "../config/config.yml",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse YAML
            self._config_data = yaml.safe_load(content)
            
            # Validate configuration
            self._validate_config()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        # Replace ${VAR_NAME} patterns
        return re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replace_env_var, content)
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['database', 'ticketing', 'email', 'monitoring']
        
        for section in required_sections:
            if section not in self._config_data:
                raise ValueError(f"Missing required configuration section: {section}")
    
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
    
    @property
    def detection(self) -> DetectionConfig:
        """Get detection configuration."""
        detection_config = self._config_data.get('detection', {})
        return DetectionConfig(
            prophet=detection_config.get('prophet', {}),
            anomaly_detection=detection_config.get('anomaly_detection', {})
        )
    
    @property
    def files(self) -> FileConfig:
        """Get file management configuration."""
        file_config = self._config_data.get('files', {})
        return FileConfig(
            chart_output_dir=file_config.get('chart_output_dir', './output/charts'),
            log_output_dir=file_config.get('log_output_dir', './logs'),
            template_dir=file_config.get('template_dir', './templates'),
            temp_dir=file_config.get('temp_dir', './temp'),
            retention=file_config.get('retention', {})
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation like 'database.username')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_enabled_kpis(self) -> List[KPIConfig]:
        """Get list of enabled KPIs for monitoring."""
        return [kpi for kpi in self.monitoring.kpis if kpi.enabled]
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.files.chart_output_dir,
            self.files.log_output_dir,
            self.files.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of configuration (excluding sensitive data)."""
        sensitive_keys = ['password', 'bearer_token', 'secret_key']
        
        def mask_sensitive_data(data: Any, path: str = '') -> Any:
            if isinstance(data, dict):
                return {
                    k: mask_sensitive_data(v, f"{path}.{k}" if path else k)
                    for k, v in data.items()
                }
            elif any(sensitive in path.lower() for sensitive in sensitive_keys):
                return "***MASKED***"
            else:
                return data
        
        masked_config = mask_sensitive_data(self._config_data)
        return yaml.dump(masked_config, default_flow_style=False)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Global configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload the global configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Reloaded configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config 