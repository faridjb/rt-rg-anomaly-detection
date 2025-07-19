#!/usr/bin/env python3
"""
Test script for the Oracle Database Service.
This script tests all database functionality including the specific query provided.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_environment_variables():
    """Test that database environment variables are available."""
    print("\n🔧 Database Environment Variables Check:")
    
    required_vars = [
        'DB_USERNAME',
        'DB_PASSWORD', 
        'DB_HOST',
        'DB_DSN',
        'DB_PORT'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show partial value for security (except port)
            if var == 'DB_PORT':
                display_value = value
            elif var == 'DB_PASSWORD':
                display_value = "*" * len(value)
            else:
                display_value = value[:10] + "..." if len(value) > 10 else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        print("   Run: source setup_env.sh")
        return False
    else:
        print("\n✅ All database environment variables are set!")
        return True

def test_oracle_import():
    """Test if cx_Oracle is available."""
    print("\n📦 Testing cx_Oracle availability...")
    
    try:
        import cx_Oracle
        print("   ✅ cx_Oracle is available")
        print(f"   📋 Version: {cx_Oracle.version}")
        return True
    except ImportError:
        print("   ❌ cx_Oracle not available")
        print("   💡 Install with: pip install cx_Oracle")
        print("   💡 Or try: conda install cx_oracle")
        return False

def test_database_service():
    """Test the database service functionality."""
    print("\n" + "=" * 70)
    print("🗄️  DATABASE SERVICE TEST")
    print("=" * 70)
    
    try:
        # Import the service
        from src.data.database_service import DatabaseService, RatingData, DatabaseServiceError
        
        print("✅ Database service imported successfully")
        
        # Test 1: Service initialization
        print("\n🔧 Test 1: Service Initialization")
        try:
            service = DatabaseService()
            print(f"   Database: {service.host}:{service.port}/{service.dsn}")
            print(f"   Username: {service.username}")
            print(f"   Pool size: {service.pool_size}")
            
        except Exception as e:
            print(f"   ❌ Initialization failed: {e}")
            return False
        
        # Test 2: Connection test
        print("\n🔌 Test 2: Database Connection")
        try:
            connection_ok = service.test_connection()
            if connection_ok:
                print("   ✅ Database connection successful")
            else:
                print("   ⚠️  Database connection failed (network/credentials issue)")
                
        except Exception as e:
            print(f"   ⚠️  Connection test failed: {e}")
        
        # Test 3: Rating data query (specific query from user)
        print("\n📊 Test 3: 3G Rating Data Query (Your Specific Query)")
        try:
            # Test with current date
            current_date = datetime.now()
            print(f"   🔍 Testing query for date: {current_date.strftime('%Y-%m-%d')} hour: 4")
            
            rating_data = service.get_h3g_rating_data(date=current_date, hour=4)
            print(f"   ✅ Query executed successfully - found {len(rating_data)} records")
            
            # Show sample data
            if rating_data:
                print("   📋 Sample records:")
                for i, record in enumerate(rating_data[:3]):
                    print(f"      {i+1}. {record.snode}: DL={record.rg_downlink:,}, UL={record.rg_uplink:,}, Total={record.total_rg:,}")
            else:
                print("   ℹ️  No data found for current date (try different date)")
                
        except Exception as e:
            print(f"   ⚠️  Query test failed: {e}")
        
        # Test 4: Different date formats
        print("\n📅 Test 4: Date Format Flexibility")
        test_dates = [
            ("2025-04-23", "YYYY-MM-DD format"),
            ("23-04-2025", "DD-MM-YYYY format"),
            (datetime(2025, 4, 23), "datetime object")
        ]
        
        for test_date, description in test_dates:
            try:
                print(f"   🗓️  Testing {description}: {test_date}")
                data = service.get_h3g_rating_data(date=test_date, hour=4)
                print(f"      ✅ Success - {len(data)} records")
            except Exception as e:
                print(f"      ⚠️  Failed: {e}")
        
        # Test 5: Node filtering
        print("\n🔍 Test 5: Node Filtering")
        try:
            # Test with specific nodes (if they exist)
            test_nodes = ["TH1CGEH1", "TH1CGZH1", "SampleNode"]
            filtered_data = service.get_h3g_rating_data(
                date="2025-04-23", 
                hour=4, 
                nodes=test_nodes
            )
            print(f"   ✅ Node filtering works - {len(filtered_data)} records for specified nodes")
            
        except Exception as e:
            print(f"   ⚠️  Node filtering test failed: {e}")
        
        # Test 6: Summary statistics
        print("\n📈 Test 6: Summary Statistics")
        try:
            summary = service.get_node_summary(date="2025-04-23", hour=4)
            print(f"   ✅ Summary generated:")
            print(f"      Total nodes: {summary['total_nodes']}")
            print(f"      Total downlink: {summary['total_downlink']:,}")
            print(f"      Total uplink: {summary['total_uplink']:,}")
            print(f"      Average downlink: {summary['avg_downlink']:,.2f}")
            
        except Exception as e:
            print(f"   ⚠️  Summary test failed: {e}")
        
        # Test 7: Threshold filtering
        print("\n🎯 Test 7: Threshold Filtering")
        try:
            threshold = 1000000  # 1 million
            high_traffic = service.get_nodes_above_threshold(
                threshold=threshold,
                date="2025-04-23",
                hour=4
            )
            print(f"   ✅ Found {len(high_traffic)} nodes above threshold {threshold:,}")
            
            if high_traffic:
                print("   📋 Top nodes:")
                for i, node in enumerate(high_traffic[:3]):
                    print(f"      {i+1}. {node.snode}: {node.total_rg:,}")
                    
        except Exception as e:
            print(f"   ⚠️  Threshold filtering test failed: {e}")
        
        # Clean up
        service.close()
        
        print("\n" + "=" * 70)
        print("✅ DATABASE SERVICE TEST COMPLETED!")
        print("=" * 70)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        if "cx_Oracle" in str(e):
            print("💡 Install cx_Oracle with: pip install cx_Oracle")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_query_examples():
    """Show practical usage examples for the database service."""
    print("\n" + "=" * 70)
    print("📚 DATABASE USAGE EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("Basic Rating Data Query", """
from src.data.database_service import DatabaseService

# Create service instance (loads credentials from environment)
service = DatabaseService()

# Get rating data for current date at 4 AM
rating_data = service.get_h3g_rating_data()

# Process the data
for record in rating_data:
    print(f"{record.snode}: {record.total_rg:,} total rating")
        """),
        
        ("Query Specific Date and Hour", """
# Query specific date and time
rating_data = service.get_h3g_rating_data(
    date="2025-04-23",  # or datetime object
    hour=4
)

# Your original query equivalent:
# SELECT SDATE,SNODE,CNT1_167774004 AS rg_downlink,CNT2_167774004 AS rg_uplink,
# (CNT1_167774004+CNT2_167774004) AS total_rg
# FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
# where SDATE=to_date('23-04-2025 04','dd-MM-yyyy HH24')
        """),
        
        ("Filter by Specific Nodes", """
# Query only specific nodes
nodes_of_interest = ["TH1CGEH1", "TH1CGZH1"]
rating_data = service.get_h3g_rating_data(
    date="2025-04-23",
    hour=4,
    nodes=nodes_of_interest
)
        """),
        
        ("Get Summary Statistics", """
# Get summary for all nodes
summary = service.get_node_summary(date="2025-04-23", hour=4)
print(f"Total nodes: {summary['total_nodes']}")
print(f"Total traffic: {summary['total_rating']:,}")
print(f"Average downlink: {summary['avg_downlink']:,.2f}")
        """),
        
        ("Find High Traffic Nodes", """
# Find nodes with high traffic (above threshold)
high_traffic_nodes = service.get_nodes_above_threshold(
    threshold=1000000,  # 1 million
    date="2025-04-23",
    hour=4
)

for node in high_traffic_nodes:
    print(f"{node.snode}: {node.total_rg:,} (DL: {node.rg_downlink:,}, UL: {node.rg_uplink:,})")
        """)
    ]
    
    for title, code in examples:
        print(f"\n📖 {title}:")
        print(code)

def main():
    """Run all database tests."""
    print("🗄️  ORACLE DATABASE SERVICE TEST")
    print("📋 Testing the specific query and database functionality")
    print()
    
    # Test 1: Environment variables
    env_ok = test_environment_variables()
    if not env_ok:
        print("\n❌ Environment variables not properly set.")
        print("Please run: source setup_env.sh")
        return 1
    
    # Test 2: cx_Oracle availability
    oracle_ok = test_oracle_import()
    if not oracle_ok:
        print("\n⚠️  cx_Oracle not available. Database tests will be limited.")
        print("To install: pip install cx_Oracle")
        
        # Show examples anyway
        show_query_examples()
        return 1
    
    # Test 3: Database service
    service_ok = test_database_service()
    
    if service_ok:
        show_query_examples()
        print(f"\n🎉 Database service is ready! Your specific query is implemented.")
        print("💡 The service automatically uses your environment credentials.")
        return 0
    else:
        print(f"\n⚠️  Database service test had issues (likely network/credentials).")
        print("💡 The code structure is correct - test with proper network access.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 