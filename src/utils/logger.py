"""
Logging utilities for the network monitoring system.

This module provides centralized logging configuration with support for:
- Multiple log handlers (console, file, rotating files)
- Component-specific loggers
- Structured logging formats
- Log rotation and retention
"""

import logging
import logging.config
import logging.handlers
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


class NetworkMonitoringLogger:
    """
    Centralized logger manager for the network monitoring system.
    
    This class provides easy access to properly configured loggers for
    different components of the system.
    """
    
    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def initialize(cls, config_path: Optional[str] = None) -> None:
        """
        Initialize logging configuration.
        
        Args:
            config_path: Path to logging configuration file
        """
        if cls._initialized:
            return
        
        config_path = config_path or cls._find_logging_config()
        
        try:
            # Create logs directory if it doesn't exist
            Path("logs").mkdir(exist_ok=True)
            
            # Load logging configuration
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Apply logging configuration
            logging.config.dictConfig(config)
            
            # Add colored formatter to console handler if running in terminal
            if sys.stdout.isatty():
                cls._add_colored_console_handler()
            
            cls._initialized = True
            
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('logs/fallback.log')
                ]
            )
            logging.error(f"Failed to load logging configuration: {e}")
    
    @classmethod
    def _find_logging_config(cls) -> str:
        """Find the logging configuration file."""
        possible_paths = [
            "config/logging.yaml",
            "config/logging.yml",
            "../config/logging.yaml",
            "../config/logging.yml",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("Logging configuration file not found")
    
    @classmethod
    def _add_colored_console_handler(cls) -> None:
        """Add colored console handler to root logger."""
        root_logger = logging.getLogger()
        
        # Find existing console handler
        console_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                console_handler = handler
                break
        
        if console_handler:
            # Replace formatter with colored one
            colored_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            console_handler.setFormatter(colored_formatter)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for the specified component.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.initialize()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def get_database_logger(cls) -> logging.Logger:
        """Get logger for database operations."""
        return cls.get_logger('src.data.database')
    
    @classmethod
    def get_ticketing_logger(cls) -> logging.Logger:
        """Get logger for ticketing operations."""
        return cls.get_logger('src.services.ticketing')
    
    @classmethod
    def get_email_logger(cls) -> logging.Logger:
        """Get logger for email operations."""
        return cls.get_logger('src.services.email_service')
    
    @classmethod
    def get_detector_logger(cls) -> logging.Logger:
        """Get logger for anomaly detection."""
        return cls.get_logger('src.core.detector')
    
    @classmethod
    def get_monitoring_logger(cls) -> logging.Logger:
        """Get logger for monitoring service."""
        return cls.get_logger('src.services.monitoring')
    
    @classmethod
    def set_level(cls, logger_name: str, level: str) -> None:
        """
        Set logging level for a specific logger.
        
        Args:
            logger_name: Name of the logger
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger = cls.get_logger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def add_handler(cls, logger_name: str, handler: logging.Handler) -> None:
        """
        Add a handler to a specific logger.
        
        Args:
            logger_name: Name of the logger
            handler: Logging handler to add
        """
        logger = cls.get_logger(logger_name)
        logger.addHandler(handler)
    
    @classmethod
    def create_file_handler(
        cls,
        filename: str,
        level: str = 'INFO',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.handlers.RotatingFileHandler:
        """
        Create a rotating file handler.
        
        Args:
            filename: Log file name
            level: Logging level
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured rotating file handler
        """
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        
        return handler


# Convenience functions for common logging operations
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return NetworkMonitoringLogger.get_logger(name)


def get_database_logger() -> logging.Logger:
    """Get database logger."""
    return NetworkMonitoringLogger.get_database_logger()


def get_ticketing_logger() -> logging.Logger:
    """Get ticketing logger."""
    return NetworkMonitoringLogger.get_ticketing_logger()


def get_email_logger() -> logging.Logger:
    """Get email logger."""
    return NetworkMonitoringLogger.get_email_logger()


def get_detector_logger() -> logging.Logger:
    """Get detector logger."""
    return NetworkMonitoringLogger.get_detector_logger()


def get_monitoring_logger() -> logging.Logger:
    """Get monitoring logger."""
    return NetworkMonitoringLogger.get_monitoring_logger()


def log_performance(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Function {func.__name__} completed in {execution_time:.2f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.2f} seconds: {e}"
            )
            raise
    
    return wrapper


def log_exception(logger: Optional[logging.Logger] = None):
    """
    Decorator to log exceptions.
    
    Args:
        logger: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Initialize logging on module import
NetworkMonitoringLogger.initialize() 