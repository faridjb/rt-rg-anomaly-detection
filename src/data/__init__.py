"""
Data module for Network Monitoring System.
Contains data models, database connections, and data access layers.
"""

# Database service imports
try:
    from .database_service import DatabaseService, RatingData, create_database_service
    __all__ = ["DatabaseService", "RatingData", "create_database_service"]
except ImportError:
    # If cx_Oracle is not available
    __all__ = []

# Add other data components as they are created 