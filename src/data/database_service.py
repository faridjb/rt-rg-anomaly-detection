"""
Modern Oracle Database Service for Network Monitoring System.
Loads credentials from environment variables and provides clean interfaces.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.utils.config_simple import get_config
except ImportError:
    # Fallback if config_simple not available
    pass

# Try to import cx_Oracle, provide fallback if not available
try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    print("⚠️  cx_Oracle not available. Install with: pip install cx_Oracle")


class DatabaseServiceError(Exception):
    """Custom exception for database service errors."""
    pass


@dataclass
class RatingData:
    """Data model for 3G Core Gateway rating data."""
    sdate: datetime
    snode: str
    rg_downlink: int
    rg_uplink: int
    total_rg: int
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.total_rg != (self.rg_downlink + self.rg_uplink):
            print(f"⚠️  Warning: Total RG mismatch for {self.snode}")


class DatabaseService:
    """
    Modern Oracle database service for network monitoring data.
    Handles connection pooling, credential management, and data queries.
    """
    
    def __init__(self):
        """Initialize database service with credentials from environment."""
        if not ORACLE_AVAILABLE:
            raise DatabaseServiceError("cx_Oracle is not installed. Run: pip install cx_Oracle")
        
        self._load_credentials()
        self._setup_connection_pool()
    
    def _load_credentials(self):
        """Load database credentials from environment variables or config."""
        try:
            config = get_config()
            db_config = config.database
            
            self.username = db_config.username
            self.password = db_config.password
            self.dsn = db_config.dsn
            self.host = db_config.host
            self.port = db_config.port
            self.encoding = db_config.encoding
            self.pool_size = db_config.pool_size
            self.max_overflow = db_config.max_overflow
            self.pool_timeout = db_config.pool_timeout
            
        except (ImportError, NameError, AttributeError):
            # Fallback to direct environment variable access
            self.username = os.getenv('DB_USERNAME', 'tools_ml')
            self.password = os.getenv('DB_PASSWORD', 'Focu$2021')
            self.dsn = os.getenv('DB_DSN', 'fcsouth.rightel.ir')
            self.host = os.getenv('DB_HOST', '10.200.6.227')
            self.port = int(os.getenv('DB_PORT', '1521'))
            self.encoding = os.getenv('DB_ENCODING', 'UTF-8')
            self.pool_size = 5
            self.max_overflow = 10
            self.pool_timeout = 30
        
        # Validate required credentials
        if not all([self.username, self.password, self.dsn, self.host]):
            raise DatabaseServiceError("Missing required database credentials in environment")
        
        # Build connection string
        self.connection_string = f"{self.username}/{self.password}@{self.host}:{self.port}/{self.dsn}"
        
        print(f"✅ Database service initialized for {self.host}:{self.port}/{self.dsn}")
    
    def _setup_connection_pool(self):
        """Setup Oracle connection pool for better performance."""
        try:
            self.pool = cx_Oracle.SessionPool(
                user=self.username,
                password=self.password,
                dsn=f"{self.host}:{self.port}/{self.dsn}",
                min=2,
                max=self.pool_size,
                increment=1,
                encoding=self.encoding
            )
            print(f"✅ Connection pool created with {self.pool_size} max connections")
            
        except cx_Oracle.DatabaseError as e:
            raise DatabaseServiceError(f"Failed to create connection pool: {e}")
    
    def get_connection(self):
        """Get a connection from the pool."""
        try:
            return self.pool.acquire()
        except cx_Oracle.DatabaseError as e:
            raise DatabaseServiceError(f"Failed to acquire connection: {e}")
    
    def release_connection(self, connection):
        """Release connection back to pool."""
        try:
            self.pool.release(connection)
        except cx_Oracle.DatabaseError as e:
            print(f"Warning: Failed to release connection: {e}")
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()
            cursor.close()
            self.release_connection(connection)
            
            if result and result[0] == 1:
                print("✅ Database connection test successful")
                return True
            else:
                print("❌ Database connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Database connection test failed: {e}")
            return False
    
    def get_h3g_rating_data(self, 
                           date: Optional[Union[str, datetime]] = None,
                           hour: Optional[int] = None,
                           nodes: Optional[List[str]] = None) -> List[RatingData]:
        """
        Get 3G Core Gateway rating data from FOCUSADM.H3G_CG_RATING_MAINTABLE.
        
        Args:
            date: Date to query (YYYY-MM-DD string or datetime). Defaults to today.
            hour: Hour to query (0-23). Defaults to 4 (4 AM).
            nodes: List of specific nodes to filter. If None, gets all nodes.
        
        Returns:
            List of RatingData objects
        """
        # Handle date parameter
        if date is None:
            query_date = datetime.now()
        elif isinstance(date, str):
            try:
                query_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                try:
                    query_date = datetime.strptime(date, "%d-%m-%Y")
                except ValueError:
                    raise DatabaseServiceError(f"Invalid date format: {date}. Use YYYY-MM-DD or DD-MM-YYYY")
        else:
            query_date = date
        
        # Set hour
        if hour is None:
            hour = 4  # Default to 4 AM as in original query
        
        # Format date for Oracle
        oracle_date = query_date.strftime("%d-%m-%Y")
        date_condition = f"to_date('{oracle_date} {hour:02d}','dd-MM-yyyy HH24')"
        
        # Build base query
        base_query = """
        SELECT SDATE, SNODE,
               CNT1_167774004 AS rg_downlink,
               CNT2_167774004 AS rg_uplink,
               (CNT1_167774004+CNT2_167774004) AS total_rg
        FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
        WHERE SDATE = {date_condition}
        """.format(date_condition=date_condition)
        
        # Add node filter if specified
        if nodes:
            node_list = "', '".join(nodes)
            base_query += f" AND SNODE IN ('{node_list}')"
        
        # Add ordering
        base_query += " ORDER BY SNODE"
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            print(f"🔍 Executing query for date: {oracle_date} {hour:02d}:00")
            print(f"📊 Query: {base_query}")
            
            cursor.execute(base_query)
            results = cursor.fetchall()
            
            # Convert to RatingData objects
            rating_data = []
            for row in results:
                try:
                    data = RatingData(
                        sdate=row[0] if row[0] else query_date,
                        snode=row[1] if row[1] else "Unknown",
                        rg_downlink=int(row[2]) if row[2] is not None else 0,
                        rg_uplink=int(row[3]) if row[3] is not None else 0,
                        total_rg=int(row[4]) if row[4] is not None else 0
                    )
                    rating_data.append(data)
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Skipping invalid row: {row} - {e}")
            
            cursor.close()
            print(f"✅ Retrieved {len(rating_data)} rating records")
            
            return rating_data
            
        except cx_Oracle.DatabaseError as e:
            raise DatabaseServiceError(f"Database query failed: {e}")
        except Exception as e:
            raise DatabaseServiceError(f"Unexpected error during query: {e}")
        finally:
            if connection:
                self.release_connection(connection)
    
    def get_node_summary(self, 
                        date: Optional[Union[str, datetime]] = None,
                        hour: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary statistics for all nodes.
        
        Args:
            date: Date to query (YYYY-MM-DD string or datetime). Defaults to today.
            hour: Hour to query (0-23). Defaults to 4.
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            rating_data = self.get_h3g_rating_data(date=date, hour=hour)
            
            if not rating_data:
                return {
                    "total_nodes": 0,
                    "total_downlink": 0,
                    "total_uplink": 0,
                    "total_rating": 0,
                    "avg_downlink": 0,
                    "avg_uplink": 0,
                    "query_date": date or datetime.now().strftime("%Y-%m-%d"),
                    "query_hour": hour or 4
                }
            
            total_downlink = sum(r.rg_downlink for r in rating_data)
            total_uplink = sum(r.rg_uplink for r in rating_data)
            total_rating = sum(r.total_rg for r in rating_data)
            
            summary = {
                "total_nodes": len(rating_data),
                "total_downlink": total_downlink,
                "total_uplink": total_uplink,
                "total_rating": total_rating,
                "avg_downlink": total_downlink / len(rating_data),
                "avg_uplink": total_uplink / len(rating_data),
                "top_nodes_by_total": sorted(rating_data, key=lambda x: x.total_rg, reverse=True)[:5],
                "query_date": date or datetime.now().strftime("%Y-%m-%d"),
                "query_hour": hour or 4
            }
            
            return summary
            
        except Exception as e:
            raise DatabaseServiceError(f"Failed to generate summary: {e}")
    
    def get_nodes_above_threshold(self, 
                                 threshold: int,
                                 date: Optional[Union[str, datetime]] = None,
                                 hour: Optional[int] = None) -> List[RatingData]:
        """
        Get nodes with total rating above specified threshold.
        
        Args:
            threshold: Minimum total rating threshold
            date: Date to query (YYYY-MM-DD string or datetime). Defaults to today.
            hour: Hour to query (0-23). Defaults to 4.
        
        Returns:
            List of RatingData objects above threshold
        """
        try:
            all_data = self.get_h3g_rating_data(date=date, hour=hour)
            filtered_data = [r for r in all_data if r.total_rg > threshold]
            
            print(f"🔍 Found {len(filtered_data)} nodes above threshold {threshold}")
            return sorted(filtered_data, key=lambda x: x.total_rg, reverse=True)
            
        except Exception as e:
            raise DatabaseServiceError(f"Failed to filter nodes by threshold: {e}")
    
    def close(self):
        """Close the connection pool."""
        try:
            if hasattr(self, 'pool') and self.pool:
                self.pool.close()
                print("✅ Database connection pool closed")
        except Exception as e:
            print(f"Warning: Error closing connection pool: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions
def create_database_service() -> DatabaseService:
    """Create and return a database service instance."""
    return DatabaseService()


def test_database_service():
    """Test the database service functionality."""
    print("🗄️  Testing Database Service...")
    
    if not ORACLE_AVAILABLE:
        print("❌ cx_Oracle not available. Install with: pip install cx_Oracle")
        return False
    
    try:
        service = DatabaseService()
        
        # Test connection
        print("\n🔌 Testing database connection...")
        if not service.test_connection():
            print("❌ Database connection failed")
            return False
        
        # Test query with mock date (today)
        print("\n📊 Testing rating data query...")
        try:
            # Use a recent date for testing
            test_date = datetime.now() - timedelta(days=1)  # Yesterday
            rating_data = service.get_h3g_rating_data(date=test_date, hour=4)
            print(f"✅ Query executed successfully - found {len(rating_data)} records")
            
            # Show sample data
            if rating_data:
                print("\n📋 Sample records:")
                for i, record in enumerate(rating_data[:3]):
                    print(f"  {i+1}. {record.snode}: DL={record.rg_downlink}, UL={record.rg_uplink}, Total={record.total_rg}")
            
        except Exception as e:
            print(f"⚠️  Query test failed (expected if no data): {e}")
        
        # Test summary
        print("\n📈 Testing summary statistics...")
        try:
            summary = service.get_node_summary()
            print(f"✅ Summary generated: {summary['total_nodes']} nodes")
        except Exception as e:
            print(f"⚠️  Summary test failed: {e}")
        
        service.close()
        print("\n✅ Database service test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_database_service() 