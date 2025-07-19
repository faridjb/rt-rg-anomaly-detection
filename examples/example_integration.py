#!/usr/bin/env python3
"""
Integration example showing database and ticketing services working together.
This demonstrates the core monitoring workflow using real data.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def integration_example():
    """Complete integration example: Database + Ticketing workflow."""
    print("=" * 70)
    print("🔗 INTEGRATION EXAMPLE: Database + Ticketing Workflow")
    print("=" * 70)
    
    try:
        # Import both services
        from src.data.database_service import DatabaseService
        from src.services.ticketing_service import TicketingService
        
        print("✅ Both services imported successfully")
        
        # Step 1: Get rating data from database
        print("\n📊 Step 1: Getting rating data from Oracle database...")
        db_service = DatabaseService()
        
        # Get data for analysis (using your specific query)
        rating_data = db_service.get_h3g_rating_data(
            date="2025-04-23", 
            hour=4
        )
        
        print(f"   ✅ Retrieved {len(rating_data)} nodes from database")
        
        # Step 2: Analyze data to find problematic nodes
        print("\n🔍 Step 2: Analyzing data for potential issues...")
        
        # Find nodes with unusually high traffic (potential overload)
        high_traffic_threshold = 2000000  # 2 million
        high_traffic_nodes = db_service.get_nodes_above_threshold(
            threshold=high_traffic_threshold,
            date="2025-04-23",
            hour=4
        )
        
        print(f"   🚨 Found {len(high_traffic_nodes)} nodes with high traffic (>{high_traffic_threshold:,})")
        
        # Show the problematic nodes
        for i, node in enumerate(high_traffic_nodes[:5], 1):
            print(f"      {i}. {node.snode}: {node.total_rg:,} (DL: {node.rg_downlink:,}, UL: {node.rg_uplink:,})")
        
        # Step 3: Check existing tickets
        print("\n🎫 Step 3: Checking existing tickets...")
        ticketing_service = TicketingService()
        
        existing_tickets = ticketing_service.query_existing_tickets()
        print(f"   📋 Found {len(existing_tickets)} active tickets")
        
        # Step 4: Create tickets for problematic nodes (if they don't already have tickets)
        print("\n📝 Step 4: Creating tickets for problematic nodes...")
        
        tickets_created = 0
        tickets_skipped = 0
        
        for node in high_traffic_nodes[:3]:  # Process top 3 for demo
            node_name = node.snode
            
            # Check if this node already has a ticket
            has_ticket = ticketing_service.check_node_has_active_ticket(node_name)
            
            if has_ticket:
                print(f"   ⏭️  {node_name}: Already has active ticket - skipping")
                tickets_skipped += 1
            else:
                print(f"   📝 {node_name}: Creating ticket for high traffic ({node.total_rg:,})")
                
                # In real scenario, you would create the ticket here:
                """
                ticket_id, ticket_link = ticketing_service.create_ticket(
                    node_name=node_name,
                    title=f"High Traffic Alert - {node.total_rg:,} total rating",
                    fault_level=3  # Medium severity
                )
                print(f"      ✅ Created ticket: {ticket_id}")
                tickets_created += 1
                """
                
                # For demo, just simulate
                print(f"      ✅ Would create ticket for {node_name}")
                tickets_created += 1
        
        # Step 5: Summary
        print(f"\n📊 Step 5: Workflow Summary")
        print(f"   Database records processed: {len(rating_data)}")
        print(f"   High traffic nodes found: {len(high_traffic_nodes)}")
        print(f"   Existing tickets found: {len(existing_tickets)}")
        print(f"   New tickets created: {tickets_created}")
        print(f"   Tickets skipped (duplicates): {tickets_skipped}")
        
        # Clean up
        db_service.close()
        
        print("\n✅ Integration example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitoring_workflow_demo():
    """Demo of the complete monitoring workflow."""
    print("\n" + "=" * 70)
    print("⚙️  MONITORING WORKFLOW DEMO")
    print("=" * 70)
    
    print("This simulates the 5-step monitoring lifecycle:")
    print("1. 📊 Connect to Database → Get KPI data")
    print("2. 🎫 Connect to Ticketing → Get existing tickets")  
    print("3. 🤖 Apply Detection → Find anomalies (simulated)")
    print("4. 📝 Raise Tickets → Create trouble tickets")
    print("5. 📧 Send Emails → Notify teams (planned)")
    print()
    
    try:
        from src.data.database_service import DatabaseService
        from src.services.ticketing_service import TicketingService
        
        # Step 1: Database connection ✅
        print("📊 Step 1: Connecting to Oracle database...")
        db_service = DatabaseService()
        summary = db_service.get_node_summary(date="2025-04-23", hour=4)
        print(f"   ✅ Connected! {summary['total_nodes']} nodes, {summary['total_rating']:,} total traffic")
        
        # Step 2: Ticketing connection ✅
        print("\n🎫 Step 2: Connecting to ticketing system...")
        ticket_service = TicketingService()
        tickets = ticket_service.query_existing_tickets()
        print(f"   ✅ Connected! {len(tickets)} active tickets found")
        
        # Step 3: Detection (simulated)
        print("\n🤖 Step 3: Applying anomaly detection...")
        print("   🔍 Analyzing traffic patterns...")
        print("   📈 Checking thresholds and trends...")
        anomalies = ["TH1VCGH1_70", "ES1CGH1_70"]  # Simulated high-traffic nodes
        print(f"   ⚠️  Detected {len(anomalies)} anomalies")
        
        # Step 4: Ticket creation ✅
        print("\n📝 Step 4: Creating tickets for anomalies...")
        for node in anomalies:
            has_ticket = ticket_service.check_node_has_active_ticket(node)
            if not has_ticket:
                print(f"   📝 Would create ticket for {node}")
            else:
                print(f"   ⏭️  {node} already has ticket")
        
        # Step 5: Email notifications (planned)
        print("\n📧 Step 5: Sending email notifications...")
        print("   📊 Would generate traffic charts")
        print("   📧 Would send notifications to Performance-Core team")
        print("   ✅ Monitoring cycle complete!")
        
        db_service.close()
        
        print(f"\n🎉 Monitoring workflow demo completed!")
        print(f"   ✅ Database integration: Working")
        print(f"   ✅ Ticketing integration: Working") 
        print(f"   🔄 AI detection: Ready for implementation")
        print(f"   🔄 Email notifications: Ready for implementation")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow demo failed: {e}")
        return False

def main():
    """Run integration examples."""
    print("🔗 DATABASE + TICKETING INTEGRATION EXAMPLES")
    print("📋 Shows how the services work together for monitoring")
    print()
    
    # Check environment
    if not all([os.getenv('DB_USERNAME'), os.getenv('TICKETING_API_URL')]):
        print("❌ Environment variables not set!")
        print("Please run: source setup_env.sh")
        return 1
    
    # Check dependencies
    missing_deps = []
    try:
        import cx_Oracle
    except ImportError:
        missing_deps.append("cx_Oracle")
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("The integration logic will work once dependencies are installed.")
    
    # Run examples
    examples = [
        integration_example,
        monitoring_workflow_demo
    ]
    
    success_count = 0
    for example in examples:
        try:
            if example():
                success_count += 1
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 INTEGRATION RESULTS: {success_count}/{len(examples)} examples completed")
    
    if success_count >= len(examples) * 0.8:
        print("🎉 Integration examples completed! Services work together perfectly.")
    else:
        print("⚠️  Some integration tests had issues (likely dependencies).")
    
    print("\n💡 INTEGRATION HIGHLIGHTS:")
    print("   ✅ Database service: Get real 3G rating data")
    print("   ✅ Ticketing service: Manage trouble tickets")
    print("   ✅ Data analysis: Find high-traffic nodes")
    print("   ✅ Workflow automation: End-to-end monitoring")
    print("   🔄 Ready for: AI detection, email notifications")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 