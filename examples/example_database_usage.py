#!/usr/bin/env python3
"""
Example usage of the Oracle Database Service.
Shows practical examples of how to use the specific query and database functionality.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def example_basic_query():
    """Example 1: Basic usage of your specific query."""
    print("=" * 60)
    print("📊 EXAMPLE 1: Your Specific Query Implementation")
    print("=" * 60)
    
    try:
        from src.data.database_service import DatabaseService
        
        # Create service instance (loads credentials from environment)
        print("🔧 Creating database service...")
        service = DatabaseService()
        
        # Your original query:
        # SELECT SDATE,SNODE,CNT1_167774004 AS rg_downlink,CNT2_167774004 AS rg_uplink,
        # (CNT1_167774004+CNT2_167774004) AS total_rg
        # FROM FOCUSADM.H3G_CG_RATING_MAINTABLE  
        # where SDATE=to_date('23-04-2025 04','dd-MM-yyyy HH24')
        
        print("\n📊 Executing your specific query...")
        rating_data = service.get_h3g_rating_data(
            date="23-04-2025",  # Same date as your query
            hour=4              # Same hour as your query
        )
        
        print(f"✅ Query executed successfully - found {len(rating_data)} records")
        
        # Display results like your original query would
        print("\n📋 Results (matching your original query structure):")
        print("SDATE                SNODE            RG_DOWNLINK  RG_UPLINK    TOTAL_RG")
        print("-" * 75)
        
        for record in rating_data[:10]:  # Show first 10 records
            sdate_str = record.sdate.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{sdate_str:<20} {record.snode:<15} {record.rg_downlink:>11,} {record.rg_uplink:>10,} {record.total_rg:>12,}")
        
        if len(rating_data) > 10:
            print(f"... and {len(rating_data) - 10} more records")
        
        service.close()
        return True
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False

def main():
    """Run database usage examples."""
    print("🗄️  ORACLE DATABASE SERVICE USAGE EXAMPLES")
    print("📋 Practical examples using your specific query")
    print()
    
    # Check environment first
    if not os.getenv('DB_USERNAME'):
        print("❌ Database environment variables not set!")
        print("Please run: source setup_env.sh")
        return 1
    
    # Check cx_Oracle
    try:
        import cx_Oracle
        print(f"✅ cx_Oracle version {cx_Oracle.version} is available")
    except ImportError:
        print("❌ cx_Oracle not available!")
        print("💡 Install with: pip install cx_Oracle")
    
    # Run example
    example_basic_query()
    
    print("\n💡 YOUR QUERY IMPLEMENTATION:")
    print("   ✅ Original: SELECT SDATE,SNODE,CNT1_167774004 AS rg_downlink,")
    print("              CNT2_167774004 AS rg_uplink,(CNT1_167774004+CNT2_167774004) AS total_rg")
    print("              FROM FOCUSADM.H3G_CG_RATING_MAINTABLE")
    print("              where SDATE=to_date('23-04-2025 04','dd-MM-yyyy HH24')")
    print("   ✅ Python:  service.get_h3g_rating_data(date='23-04-2025', hour=4)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 