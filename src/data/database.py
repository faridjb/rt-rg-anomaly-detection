"""
Database connection and query module for the network monitoring system.

This module provides:
- Oracle database connection management with pooling
- Structured KPI data queries
- Connection health monitoring
- Query optimization and caching
"""

import cx_Oracle
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import time
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from ..utils.config import get_config
from ..utils.logger import get_database_logger, log_performance, log_exception
from .models import KPIData, KPIQuery, LocationInfo, dataframe_to_kpi_data


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class ConnectionPool:
    """Oracle database connection pool manager."""
    
    def __init__(self, config):
        """
        Initialize connection pool.
        
        Args:
            config: Database configuration object
        """
        self.config = config
        self.logger = get_database_logger()
        self._pool = None
        self._engine = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the database connection pool."""
        try:
            # Create DSN
            dsn_tns = cx_Oracle.makedsn(
                self.config.host, 
                self.config.port, 
                service_name=self.config.dsn
            )
            
            # Create SQLAlchemy engine with connection pooling
            connection_string = (
                f"oracle+cx_oracle://{self.config.username}:"
                f"{self.config.password}@{dsn_tns}"
            )
            
            self._engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=3600,  # Recycle connections every hour
                pool_pre_ping=True,  # Validate connections before use
                echo=False  # Set to True for SQL debugging
            )
            
            self.logger.info(
                f"Database connection pool initialized: "
                f"pool_size={self.config.pool_size}, "
                f"max_overflow={self.config.max_overflow}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Failed to initialize connection pool: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.
        
        Yields:
            Database connection
        """
        connection = None
        try:
            connection = self._engine.connect()
            yield connection
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1 FROM DUAL"))
                row = result.fetchone()
                return row[0] == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status information.
        
        Returns:
            Dictionary with pool status metrics
        """
        if self._engine and hasattr(self._engine.pool, 'status'):
            pool = self._engine.pool
            return {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
        return {}


class NetworkDatabase:
    """
    Main database interface for network monitoring data.
    
    This class provides high-level methods for querying network KPI data
    and managing database connections.
    """
    
    def __init__(self, config=None):
        """
        Initialize database interface.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config or get_config()
        self.logger = get_database_logger()
        self.pool = ConnectionPool(self.config.database)
        self._query_cache = {}
    
    @log_performance
    @log_exception()
    def fetch_kpi_data(self, query: KPIQuery) -> List[KPIData]:
        """
        Fetch KPI data based on query parameters.
        
        Args:
            query: KPI query parameters
            
        Returns:
            List of KPIData objects
        """
        self.logger.info(
            f"Fetching KPI data: {query.kpi_name} "
            f"from {query.start_time} to {query.end_time}"
        )
        
        # Get the appropriate SQL query for the KPI
        sql_query = self._get_kpi_query(query)
        
        try:
            with self.pool.get_connection() as conn:
                # Execute query and get DataFrame
                df = pd.read_sql(sql_query, conn)
                
                if df.empty:
                    self.logger.warning(f"No data found for KPI: {query.kpi_name}")
                    return []
                
                # Sort by date
                df = df.sort_values(['SDATE'], ascending=True)
                
                # Convert to KPIData objects
                kpi_data = dataframe_to_kpi_data(df, query.kpi_name)
                
                self.logger.info(
                    f"Retrieved {len(kpi_data)} records for {query.kpi_name}"
                )
                
                return kpi_data
                
        except Exception as e:
            self.logger.error(f"Failed to fetch KPI data: {e}")
            raise DatabaseError(f"Failed to fetch KPI data: {e}")
    
    def _get_kpi_query(self, query: KPIQuery) -> str:
        """
        Get SQL query for specific KPI type.
        
        Args:
            query: KPI query parameters
            
        Returns:
            SQL query string
        """
        start_time_str = query.start_time.strftime('%d-%m-%Y %H')
        end_time_str = query.end_time.strftime('%d-%m-%Y %H')
        
        # Build WHERE clause
        where_clause = f"""
            WHERE SDATE BETWEEN 
                to_date('{start_time_str}', 'dd-MM-yyyy HH24') AND
                to_date('{end_time_str}', 'dd-MM-yyyy HH24')
        """
        
        if query.snode_filter:
            where_clause += f" AND SNODE = '{query.snode_filter}'"
        
        # KPI-specific queries
        if query.kpi_name == "3G Traffic Channel Assignment Success Rate":
            return f"""
                SELECT SDATE, SNODE,
                    (H.CNT27_83888176 / DECODE(H.CNT29_83888176, 0, null, H.CNT29_83888176)) * 100 
                    AS "3G Traffic Channel Assignment",
                    H.CNT29_83888176 AS "Assignment Request Times"
                FROM FOCUSADM.H3G_MSS_RNC_MAINTABLE H
                {where_clause}
                ORDER BY SDATE
            """
        
        elif query.kpi_name == "Paging Success Rate":
            return f"""
                SELECT SDATE, SNODE, 
                    ROUND(DECODE(CNT57_83888258, 0, 0, (CNT56_83888258 / CNT57_83888258) * 100), 2)
                    AS "Paging success rate",
                    CNT57_83888258 AS ATTEMPT
                FROM FOCUSADM.H3G_SAI_MAINTABLE
                {where_clause}
                ORDER BY SDATE
            """
        
        elif query.kpi_name == "APN Traffic":
            return f"""
                SELECT SDATE, SNODE,
                    CNT3_134221235 AS "GTPv1 AR",
                    CNT11_134221235 AS "GTPv1 SR",
                    CNT6_134221244 AS "Gi DL",
                    CNT9_134221244 AS "Gi UL",
                    CNT3_140509186 AS "GW Inc DLT",
                    CNT33_134221244 AS "GGSN DL discard",
                    CNT35_134221244 AS "GGSN UL discard",
                    CNT17_134221235 AS "GTPv1 Aut Fail",
                    CNT37_134221235 AS "GTPv1 Sys Fault",
                    CNT17_134221237 AS "PDP Act SR",
                    CNT40_134221235 AS "unknown addr or type",
                    CNT36_134221235 AS "service not support",
                    CNT31_134221235 AS "OCS Server No Response",
                    CNT29_134221235 AS "no resource"
                FROM FOCUSADM.H3G_APN_MAINTABLE
                {where_clause}
                ORDER BY SDATE
            """
        
        elif query.kpi_name == "Incoming Trunk Office":
            return f"""
                SELECT SDATE, SNODE,
                    ROUND(CNT4_83888300, 2) AS "IN_AR",
                    ROUND(CNT5_83888300, 2) AS "IN_AT",
                    ROUND(CNT6_83888300, 2) AS "IN_ATR",
                    ROUND(CNT11_83888300, 2) AS "IN_CALL_CR",
                    ROUND(CNT48_83888300, 2) AS "IN_SZ_T",
                    ROUND(CNT22_83888300, 2) AS "IN_CNG_T",
                    ROUND(CNT35_83888300, 2) AS "IN_LO_FAIL",
                    ROUND(CNT44_83888300, 2) AS "IN_O_FAIL_CAUSES",
                    CNT1_83888300 AS "Address Invalid Times",
                    CNT26_83888300 AS "Interworking Failures",
                    CNT39_83888300 AS "Number of Available circuits",
                    CNT63_83888300 AS "Number of fault circuits"
                FROM FOCUSADM.H3G_OFFICES_MAINTABLE
                {where_clause}
                ORDER BY SDATE
            """
        
        else:
            raise ValueError(f"Unsupported KPI type: {query.kpi_name}")
    
    @log_performance
    def get_node_location(self, snode: str) -> Optional[LocationInfo]:
        """
        Get location information for a network node.
        
        Args:
            snode: Network node identifier
            
        Returns:
            LocationInfo object or None if not found
        """
        query = """
            SELECT RNCNAME, RNCID, CITY, PROVINCE
            FROM RIGHTELADM_TEMP.Rightel_Cell_Db
            WHERE RNCNAME = :snode
        """
        
        try:
            with self.pool.get_connection() as conn:
                result = conn.execute(text(query), {'snode': snode})
                row = result.fetchone()
                
                if row:
                    return LocationInfo(
                        snode=row[0],
                        city=row[2],
                        province=row[3]
                    )
                else:
                    self.logger.warning(f"No location found for node: {snode}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get node location: {e}")
            return None
    
    @log_performance
    def get_active_nodes(self, hours_back: int = 24) -> List[str]:
        """
        Get list of active network nodes within specified time range.
        
        Args:
            hours_back: Number of hours to look back for activity
            
        Returns:
            List of active node identifiers
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.strftime('%d-%m-%Y %H')
        
        # Query to get active nodes from multiple tables
        queries = [
            f"""
                SELECT DISTINCT SNODE FROM FOCUSADM.H3G_MSS_RNC_MAINTABLE
                WHERE SDATE >= to_date('{cutoff_str}', 'dd-MM-yyyy HH24')
            """,
            f"""
                SELECT DISTINCT SNODE FROM FOCUSADM.H3G_SAI_MAINTABLE
                WHERE SDATE >= to_date('{cutoff_str}', 'dd-MM-yyyy HH24')
            """,
            f"""
                SELECT DISTINCT SNODE FROM FOCUSADM.H3G_APN_MAINTABLE
                WHERE SDATE >= to_date('{cutoff_str}', 'dd-MM-yyyy HH24')
            """,
            f"""
                SELECT DISTINCT SNODE FROM FOCUSADM.H3G_OFFICES_MAINTABLE
                WHERE SDATE >= to_date('{cutoff_str}', 'dd-MM-yyyy HH24')
            """
        ]
        
        active_nodes = set()
        
        try:
            with self.pool.get_connection() as conn:
                for query in queries:
                    try:
                        result = conn.execute(text(query))
                        nodes = [row[0] for row in result.fetchall()]
                        active_nodes.update(nodes)
                    except Exception as e:
                        self.logger.warning(f"Query failed: {e}")
                        continue
            
            nodes_list = sorted(list(active_nodes))
            self.logger.info(f"Found {len(nodes_list)} active nodes")
            return nodes_list
            
        except Exception as e:
            self.logger.error(f"Failed to get active nodes: {e}")
            return []
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get database connection status and metrics.
        
        Returns:
            Dictionary with connection status information
        """
        status = {
            'connected': False,
            'pool_status': {},
            'last_test_time': None,
            'test_result': False
        }
        
        try:
            test_start = time.time()
            status['test_result'] = self.pool.test_connection()
            status['connected'] = status['test_result']
            status['last_test_time'] = datetime.now()
            status['test_duration_ms'] = (time.time() - test_start) * 1000
            status['pool_status'] = self.pool.get_pool_status()
            
        except Exception as e:
            self.logger.error(f"Failed to get connection status: {e}")
            status['error'] = str(e)
        
        return status
    
    def execute_custom_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            pandas DataFrame with query results
        """
        self.logger.info("Executing custom query")
        
        try:
            with self.pool.get_connection() as conn:
                if params:
                    df = pd.read_sql(text(query), conn, params=params)
                else:
                    df = pd.read_sql(text(query), conn)
                
                self.logger.info(f"Custom query returned {len(df)} rows")
                return df
                
        except Exception as e:
            self.logger.error(f"Custom query failed: {e}")
            raise DatabaseError(f"Custom query failed: {e}")
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if self.pool._engine:
                self.pool._engine.dispose()
                self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Global database instance
_database: Optional[NetworkDatabase] = None


def get_database(config=None) -> NetworkDatabase:
    """
    Get the global database instance.
    
    Args:
        config: Configuration object (optional)
        
    Returns:
        Global database instance
    """
    global _database
    if _database is None:
        _database = NetworkDatabase(config)
    return _database


def close_database() -> None:
    """Close the global database instance."""
    global _database
    if _database is not None:
        _database.close()
        _database = None 