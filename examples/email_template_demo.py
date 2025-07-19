#!/usr/bin/env python3
"""
SG/RG Interface Traffic Alert Email Template Demo

This script demonstrates how to use the modern email template
for sending traffic anomaly alerts in telecom networks.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.email_service import EmailService, create_sample_alert_data


def create_critical_alert_data():
    """Create critical traffic alert data example."""
    return {
        'interface_name': 'SG-CORE-TH1-GE-0/0/1',
        'node_name': 'TH1VCGH1_70',
        'detection_time': '2024-12-19 14:30:25',
        'anomaly_type': 'Traffic Surge - Capacity Overload',
        'severity_level': 'CRITICAL',
        'ticket_number': 'TT-2024-001234',
        'ticket_link': 'https://10.201.6.13/KM_UCMS_TT/ticket/TT-2024-001234',
        'current_traffic_gbps': '8.5',
        'threshold_exceeded_percent': '185',
        'duration_minutes': '45',
        'affected_users': '12,500',
        'current_traffic': '8.5',
        'baseline_traffic': '4.2',
        'threshold_traffic': '6.0',
        'utilization_percent': '95',
        'traffic_chart_placeholder': '📈 Traffic spike from 4.2 to 8.5 Gbps at 14:30',
        'performance_chart_placeholder': '📊 Packet loss: 2.1%, Latency: +45ms',
        'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'system_id': 'CORE-AI-SG-001'
    }


def create_high_alert_data():
    """Create high priority alert data example."""
    return {
        'interface_name': 'RG-CORE-ES1-GE-0/1/2',
        'node_name': 'ES1CGH1_101',
        'detection_time': '2024-12-19 15:45:12',
        'anomaly_type': 'Sustained High Traffic',
        'severity_level': 'HIGH',
        'ticket_number': 'TT-2024-001235',
        'ticket_link': 'https://10.201.6.13/KM_UCMS_TT/ticket/TT-2024-001235',
        'current_traffic_gbps': '7.2',
        'threshold_exceeded_percent': '120',
        'duration_minutes': '120',
        'affected_users': '8,200',
        'current_traffic': '7.2',
        'baseline_traffic': '5.8',
        'threshold_traffic': '6.0',
        'utilization_percent': '86',
        'traffic_chart_placeholder': '📈 Steady increase from 5.8 to 7.2 Gbps over 2 hours',
        'performance_chart_placeholder': '📊 Response time degradation, QoS metrics declining',
        'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'system_id': 'CORE-AI-RG-002'
    }


def create_medium_alert_data():
    """Create medium priority alert data example."""
    return {
        'interface_name': 'SG-ACCESS-BP1-FE-0/0/5',
        'node_name': 'BP1CGHZ_45',
        'detection_time': '2024-12-19 16:20:08',
        'anomaly_type': 'Traffic Pattern Anomaly',
        'severity_level': 'MEDIUM',
        'ticket_number': 'TT-2024-001236',
        'ticket_link': 'https://10.201.6.13/KM_UCMS_TT/ticket/TT-2024-001236',
        'current_traffic_gbps': '3.8',
        'threshold_exceeded_percent': '110',
        'duration_minutes': '25',
        'affected_users': '2,100',
        'current_traffic': '3.8',
        'baseline_traffic': '3.2',
        'threshold_traffic': '3.5',
        'utilization_percent': '76',
        'traffic_chart_placeholder': '📈 Unusual traffic pattern detected',
        'performance_chart_placeholder': '📊 Minor KPI fluctuations observed',
        'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'system_id': 'CORE-AI-SG-003'
    }


def demo_email_templates():
    """Demonstrate different email templates."""
    print("🎨 SG/RG Interface Traffic Alert Email Template Demo")
    print("=" * 60)
    
    # Initialize email service
    email_service = EmailService()
    
    # Demo scenarios
    scenarios = [
        ("CRITICAL Traffic Surge", create_critical_alert_data()),
        ("HIGH Sustained Traffic", create_high_alert_data()),
        ("MEDIUM Pattern Anomaly", create_medium_alert_data())
    ]
    
    for i, (name, alert_data) in enumerate(scenarios, 1):
        print(f"\n📧 {i}. {name}")
        print("-" * 40)
        
        try:
            # Create email
            message = email_service.create_sg_rg_alert_email(
                alert_data=alert_data,
                recipients=['performance-team@rightel.ir'],
                cc_recipients=['noc@rightel.ir']
            )
            
            # Generate preview
            html_content = email_service.format_sg_rg_alert_template(alert_data)
            
            # Save preview
            preview_dir = Path('temp')
            preview_dir.mkdir(exist_ok=True)
            
            severity = alert_data['severity_level'].lower()
            interface = alert_data['interface_name'].replace('/', '_').replace('-', '_')
            preview_file = preview_dir / f"email_preview_{severity}_{interface}.html"
            
            with open(preview_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Email created successfully")
            print(f"📋 Subject: {message['Subject']}")
            print(f"🎯 Interface: {alert_data['interface_name']}")
            print(f"🔥 Severity: {alert_data['severity_level']}")
            print(f"🎫 Ticket: {alert_data['ticket_number']}")
            print(f"💾 Preview: {preview_file}")
            
        except Exception as e:
            print(f"❌ Error creating email: {e}")
    
    print(f"\n🌐 Open the preview files in your browser to see the templates!")
    print(f"📁 Preview files location: {Path('temp').absolute()}")


def demo_real_world_usage():
    """Show real-world usage example."""
    print("\n\n🔧 Real-World Usage Example")
    print("=" * 60)
    
    print("""
# Real implementation example:

from services.email_service import EmailService
from services.ticketing_service import TicketingService
from data.database_service import DatabaseService

# 1. Detect anomaly (from your AI detection)
anomaly = detect_traffic_anomaly('SG-CORE-01-GE-0/0/1')

# 2. Create ticket
ticketing = TicketingService()
ticket_id, ticket_link = ticketing.create_ticket(
    node_name='TH1VCGH1_70',
    title=f'Traffic Anomaly: {anomaly.interface}',
    fault_level=3
)

# 3. Prepare alert data
alert_data = {
    'interface_name': anomaly.interface,
    'node_name': anomaly.node,
    'detection_time': anomaly.timestamp,
    'anomaly_type': anomaly.type,
    'severity_level': anomaly.severity,
    'ticket_number': ticket_id,
    'ticket_link': ticket_link,
    'current_traffic_gbps': str(anomaly.current_value),
    # ... other metrics
}

# 4. Send email alert
email_service = EmailService()
success = email_service.send_sg_rg_alert(
    alert_data=alert_data,
    recipients=['performance-core@rightel.ir'],
    cc_recipients=['noc@rightel.ir']
)

if success:
    print("📧 Alert sent successfully!")
""")


if __name__ == "__main__":
    # Run the demos
    demo_email_templates()
    demo_real_world_usage()
    
    print("\n🎉 Demo completed!")
    print("💡 The email templates are ready for production use.")
    print("🔗 Integration with ticketing and database services is seamless.") 