"""
Services module for Network Monitoring System.
Contains business logic services for ticketing, email, and monitoring.
"""

# Service imports will be added as services are created
try:
    from .ticketing_service import TicketingService, create_ticketing_service
    __all__ = ["TicketingService", "create_ticketing_service"]
except ImportError:
    __all__ = [] 