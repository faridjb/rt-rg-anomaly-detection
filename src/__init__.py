"""
Network Performance AI Anomaly Detection System

A modern, production-ready system for network monitoring with:
- Oracle database integration for 3G rating data
- Automated ticketing system integration  
- AI-based anomaly detection
- Email notifications with charts
- Comprehensive health monitoring
"""

__version__ = "1.0.0"

# Import working modules
try:
    from .utils.config_simple import get_config
    from .utils.logger import get_logger
    from .services.ticketing_service import TicketingService, create_ticketing_service
    from .data.database_service import DatabaseService, RatingData, create_database_service
    
    __all__ = [
        "get_config", 
        "get_logger", 
        "TicketingService", 
        "create_ticketing_service",
        "DatabaseService",
        "RatingData", 
        "create_database_service"
    ]
    
except ImportError as e:
    # Graceful fallback if dependencies are missing
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = [] 