"""
Modern Ticketing Service for UCMS TT System.
Loads credentials from environment variables and provides clean interfaces.
"""

import json
import os
import requests
import urllib3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.utils.config_simple import get_config
except ImportError:
    # Fallback if config_simple not available
    pass

# Disable SSL warnings for internal network
urllib3.disable_warnings()


class TicketingServiceError(Exception):
    """Custom exception for ticketing service errors."""
    pass


class TicketingService:
    """
    Modern ticketing service for UCMS TT system.
    Handles ticket creation, querying, and file attachments.
    """
    
    def __init__(self):
        """Initialize ticketing service with credentials from environment."""
        self._load_credentials()
        self._setup_session()
    
    def _load_credentials(self):
        """Load credentials from environment variables or config."""
        try:
            config = get_config()
            ticketing_config = config.ticketing
            
            self.api_url = ticketing_config.api_url
            self.username = ticketing_config.username
            self.password = ticketing_config.password
            self.bearer_token = ticketing_config.bearer_token
            self.timeout = ticketing_config.timeout
            self.retry_attempts = ticketing_config.retry_attempts
            
        except (ImportError, NameError):
            # Fallback to direct environment variable access
            self.api_url = os.getenv('TICKETING_API_URL', 'https://10.201.6.13/KM_UCMS_TT')
            self.username = os.getenv('TICKETING_USERNAME', 'EX.F.Jabarimaleki')
            self.password = os.getenv('TICKETING_PASSWORD', '7s&aBvP/#49.qjC7')
            self.bearer_token = os.getenv('TICKETING_BEARER_TOKEN', 
                'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJLTV9VQ01TIiwiaWF0IjoxNjQxMjAwOTQ3LCJleHAiOjIzMDM4ODg5NDcsImF1ZCI6Ind3dy5rbS1lbC5pciIsInN1YiI6ImluZm9Aa20tZWwuaXIifQ.3e3MuRM_TktHSwi7Vitbj5N0zR0E1Dg2H_t1T1zE0ZY')
            self.timeout = 30
            self.retry_attempts = 3
        
        # Validate required credentials
        if not all([self.api_url, self.username, self.password, self.bearer_token]):
            raise TicketingServiceError("Missing required ticketing credentials in environment")
        
        # File upload URL
        self.file_upload_url = "https://10.201.6.13/KM_UCMS_TT_FILES"
        
        print(f"✅ Ticketing service initialized with API: {self.api_url}")
    
    def _setup_session(self):
        """Setup requests session with common headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        })
        self.session.verify = False  # Disable SSL verification for internal network
    
    def _make_request(self, method: str, url: str, data: Dict = None, 
                     files: Dict = None, **kwargs) -> requests.Response:
        """Make HTTP request with error handling and retries."""
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data if files is None else None,
                    data=data if files is not None else None,
                    files=files,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise TicketingServiceError(f"Request failed after {self.retry_attempts} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying...")
        
        raise TicketingServiceError("Request failed")
    
    def get_api_specifications(self, field_type: str, vendor: Optional[str] = None) -> List[Dict]:
        """
        Get API specifications for various fields.
        
        Args:
            field_type: Type of field ("Fault Level", "Department", "Trouble Source", 
                       "Users Groups", "Domain", "NE Name")
            vendor: Vendor type ("ZTE", "Huawei") - required for Domain and NE Name
        
        Returns:
            List of available options for the specified field
        """
        request_data = {
            "username": self.username,
            "password": self.password,
            "type": field_type
        }
        
        # Add vendor-specific parameters
        if field_type in ["Domain", "NE Name"] and vendor:
            request_data["Department"] = vendor
        
        try:
            response = self._make_request("POST", self.api_url, data=request_data)
            result = response.json()
            
            if 'Data' in result and len(result['Data']) > 0:
                return result['Data'][0]
            else:
                return []
                
        except Exception as e:
            raise TicketingServiceError(f"Failed to get API specifications for {field_type}: {e}")
    
    def query_existing_tickets(self, date_from: str = "2022-01-01", 
                              date_to: Optional[str] = None,
                              title_filter: str = "(Auto Ticketing System-Core)",
                              domain: str = "CS",
                              status: str = "Running") -> List[Tuple[str, str]]:
        """
        Query existing trouble tickets.
        
        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD), defaults to today
            title_filter: Filter by title
            domain: Domain filter
            status: Ticket status filter
        
        Returns:
            List of tuples (description, ticket_id)
        """
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        query_data = {
            "username": self.username,
            "password": self.password,
            "type": "Query Trouble Ticket",
            "Title": title_filter,
            "Ticket_Time_F": date_from,
            "Ticket_Time_T": date_to,
            "Fault_First_Occur_Time_F": date_from,
            "Fault_First_Occur_Time_T": date_to,
            "SiteID": "",
            "NE_NAME": "",
            "Domain": domain,
            "Description": "",
            "Ticket_Status": status
        }
        
        try:
            response = self._make_request("POST", self.api_url, data=query_data)
            result = response.json()
            
            existing_tickets = []
            if 'Data' in result and len(result['Data']) > 0 and result['Data'][0]:
                for ticket in result['Data'][0]:
                    existing_tickets.append((
                        ticket['Description'],
                        ticket['TicketId']
                    ))
            
            print(f"Found {len(existing_tickets)} existing tickets")
            return existing_tickets
            
        except Exception as e:
            raise TicketingServiceError(f"Failed to query existing tickets: {e}")
    
    def create_ticket(self, node_name: str, title: str, 
                     fault_occur_time: Optional[str] = None,
                     department: Optional[str] = None,
                     fault_level: int = 3,
                     domain: str = "CS",
                     affected_service: int = 1) -> Tuple[str, str]:
        """
        Create a new trouble ticket.
        
        Args:
            node_name: Name of the affected node
            title: Ticket title
            fault_occur_time: When the fault occurred (YYYY-MM-DD HH:MM:SS)
            department: Department/vendor (auto-detected from node name if not provided)
            fault_level: Severity level (1-5)
            domain: Domain (default: CS)
            affected_service: Service affected flag (0/1)
        
        Returns:
            Tuple of (ticket_number, ticket_link)
        """
        # Auto-detect department from node name if not provided
        if department is None:
            if node_name.endswith('H') or 'H' in node_name[-3:]:
                department = 'Huawei'
            elif node_name.endswith('Z') or 'Z' in node_name[-3:]:
                department = 'ZTE'
            else:
                department = 'Huawei'  # Default
        
        # Use current time if fault time not provided
        if fault_occur_time is None:
            fault_occur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ticket_data = {
            "username": self.username,
            "password": self.password,
            "type": "New Trouble Ticket",
            "Department": department,
            "Trouble_Source": "NPM-CS",
            "Fault_First_Occur_Time": fault_occur_time,
            "Fault_Level": fault_level,
            "Domain": domain,
            "AlarmName": "Other_CS",
            "NE_Name_Site_ID": node_name,
            "Site_Province": "",
            "Region": "",
            "Title": f"(Auto Ticketing System-Core){title}",
            "Down_Site_Count": 0,
            "Affected_Site_List": "",
            "Description": node_name,
            "Service_Affected": affected_service,
            "Is_Site_Down": 0,
            "Pending_For": 10001466,  # Performance team
            "Assign_To": 10001466     # API account
        }
        
        try:
            response = self._make_request("POST", self.api_url, data=ticket_data)
            
            # Extract ticket number from response
            response_text = response.text
            # The ticket number is typically in a specific position in the response
            ticket_number = response_text[98:119].strip() if len(response_text) > 119 else "Unknown"
            
            # Clean up ticket number (remove any non-alphanumeric characters except hyphens)
            import re
            ticket_number = re.sub(r'[^A-Za-z0-9\-]', '', ticket_number)
            
            ticket_link = f'https://10.201.6.13/alL_T?ticket_id={ticket_number}'
            
            print(f"✅ Ticket created successfully for {node_name} -> TT: {ticket_number}")
            return ticket_number, ticket_link
            
        except Exception as e:
            raise TicketingServiceError(f"Failed to create ticket for {node_name}: {e}")
    
    def check_node_has_active_ticket(self, node_name: str) -> bool:
        """
        Check if a node already has an active ticket.
        
        Args:
            node_name: Name of the node to check
        
        Returns:
            True if node has active ticket, False otherwise
        """
        try:
            existing_tickets = self.query_existing_tickets()
            
            for description, ticket_id in existing_tickets:
                if node_name in description:
                    print(f"Node {node_name} already has active ticket: {ticket_id}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Warning: Could not check existing tickets for {node_name}: {e}")
            return False


# Convenience functions
def create_ticketing_service() -> TicketingService:
    """Create and return a ticketing service instance."""
    return TicketingService()


def test_ticketing_service():
    """Test the ticketing service functionality."""
    print("🎫 Testing Ticketing Service...")
    
    try:
        service = TicketingService()
        
        # Test querying existing tickets
        print("\n📋 Querying existing tickets...")
        tickets = service.query_existing_tickets()
        print(f"Found {len(tickets)} active tickets")
        
        # Test getting API specs
        print("\n📊 Testing API specifications...")
        departments = service.get_api_specifications("Department")
        print(f"Available departments: {len(departments)} found")
        
        print("\n✅ Ticketing service test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ticketing service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_ticketing_service() 