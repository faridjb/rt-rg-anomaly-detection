"""
Email service for sending SG/RG interface traffic alerts.

This module provides functionality to generate and send professional
email notifications for network traffic anomalies.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending network performance alerts via email."""
    
    def __init__(self) -> None:
        """Initialize email service with environment configuration."""
        self.smtp_host = os.getenv('EMAIL_SMTP_HOST', 'mail.rightel.ir')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '465'))
        self.username = os.getenv('EMAIL_USERNAME', 'Performance-Dev')
        self.password = os.getenv('EMAIL_PASSWORD', 'Focus2021')
        self.use_tls = os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'
        
        # Default recipients
        self.default_recipients = [
            'Performance-Core@rightel.ir',
            'Performance-Tools@rightel.ir'
        ]
        
        # Template paths
        self.template_dir = Path(__file__).parent.parent.parent / 'templates' / 'email_templates'
        
    def load_template(self, template_name: str) -> str:
        """Load HTML email template from file."""
        template_path = self.template_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
            
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def format_sg_rg_alert_template(self, alert_data: Dict[str, Any]) -> str:
        """Format the SG/RG interface traffic alert template with data."""
        template = self.load_template('sg_rg_traffic_alert.html')
        
        # Default values for missing data
        defaults = {
            'interface_name': 'Unknown Interface',
            'node_name': 'Unknown Node',
            'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anomaly_type': 'Traffic Spike',
            'severity_level': 'HIGH',
            'ticket_number': 'TT-XXXX-XXXX',
            'ticket_link': '#',
            'current_traffic_gbps': '0.0',
            'threshold_exceeded_percent': '0',
            'duration_minutes': '0',
            'affected_users': '0',
            'current_traffic': '0.0',
            'baseline_traffic': '0.0',
            'threshold_traffic': '0.0',
            'utilization_percent': '0',
            'traffic_chart_placeholder': '📈 Chart will be generated here',
            'performance_chart_placeholder': '📊 Performance metrics chart',
            'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'system_id': 'CORE-AI-001'
        }
        
        # Merge provided data with defaults
        formatted_data = {**defaults, **alert_data}
        
        # Format the template
        try:
            formatted_html = template.format(**formatted_data)
            return formatted_html
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template formatting error: missing variable {e}")
    
    def create_sg_rg_alert_email(
        self,
        alert_data: Dict[str, Any],
        recipients: Optional[List[str]] = None,
        cc_recipients: Optional[List[str]] = None
    ) -> MIMEMultipart:
        """Create a formatted email message for SG/RG interface alerts."""
        
        # Use default recipients if none provided
        if recipients is None:
            recipients = self.default_recipients
            
        # Create email message
        msg = MIMEMultipart('alternative')
        
        # Email headers
        interface_name = alert_data.get('interface_name', 'Unknown')
        severity = alert_data.get('severity_level', 'HIGH')
        
        msg['Subject'] = f"🚨 {severity} Alert: SG/RG Interface Traffic Anomaly - {interface_name}"
        msg['From'] = f"Network AI System <{self.username}@rightel.ir>"
        msg['To'] = ', '.join(recipients)
        
        if cc_recipients:
            msg['Cc'] = ', '.join(cc_recipients)
        
        # Add reply-to
        msg['Reply-To'] = 'Performance-Tools@rightel.ir'
        
        # Add priority headers for high severity
        if severity.upper() in ['HIGH', 'CRITICAL']:
            msg['X-Priority'] = '1'
            msg['X-MSMail-Priority'] = 'High'
            msg['Importance'] = 'High'
        
        # Format HTML content
        html_content = self.format_sg_rg_alert_template(alert_data)
        
        # Create plain text version for compatibility
        plain_text = self._html_to_plain_text(alert_data)
        
        # Attach both versions
        msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))
        
        return msg
    
    def _html_to_plain_text(self, alert_data: Dict[str, Any]) -> str:
        """Create a plain text version of the alert."""
        interface_name = alert_data.get('interface_name', 'Unknown')
        node_name = alert_data.get('node_name', 'Unknown')
        detection_time = alert_data.get('detection_time', 'Unknown')
        severity = alert_data.get('severity_level', 'HIGH')
        ticket_number = alert_data.get('ticket_number', 'TT-XXXX-XXXX')
        
        return f"""
SG/RG INTERFACE TRAFFIC ALERT - {severity}
========================================

ALERT DETAILS:
- Interface: {interface_name}
- Node: {node_name}
- Detection Time: {detection_time}
- Severity: {severity}
- Ticket: {ticket_number}

TRAFFIC METRICS:
- Current Traffic: {alert_data.get('current_traffic_gbps', '0.0')} Gbps
- Threshold Exceeded: {alert_data.get('threshold_exceeded_percent', '0')}%
- Duration: {alert_data.get('duration_minutes', '0')} minutes
- Affected Users: {alert_data.get('affected_users', '0')}

This is an automated alert from the Network Performance AI System.
For details, please view the HTML version or contact Performance-Tools@rightel.ir

---
Rightel Communications - Network Operations Center
        """.strip()
    
    def send_email(
        self,
        message: MIMEMultipart,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Send email message via SMTP."""
        if recipients is None:
            recipients = self.default_recipients
            
        try:
            # Create SMTP connection
            if self.use_tls:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            
            # Login and send
            server.login(self.username, self.password)
            
            # Get all recipients (To + Cc)
            all_recipients = recipients.copy()
            if message.get('Cc'):
                cc_list = [addr.strip() for addr in message['Cc'].split(',')]
                all_recipients.extend(cc_list)
            
            # Send email
            server.send_message(message, to_addrs=all_recipients)
            server.quit()
            
            logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_sg_rg_alert(
        self,
        alert_data: Dict[str, Any],
        recipients: Optional[List[str]] = None,
        cc_recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send SG/RG interface traffic alert email.
        
        Args:
            alert_data: Dictionary containing alert information
            recipients: List of primary recipients
            cc_recipients: List of CC recipients
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            # Create email message
            message = self.create_sg_rg_alert_email(
                alert_data=alert_data,
                recipients=recipients,
                cc_recipients=cc_recipients
            )
            
            # Send email
            return self.send_email(message, recipients)
            
        except Exception as e:
            logger.error(f"Failed to send SG/RG alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP server connection."""
        try:
            if self.use_tls:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            
            server.login(self.username, self.password)
            server.quit()
            
            logger.info("SMTP connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False


# Example usage and demo
def create_sample_alert_data() -> Dict[str, Any]:
    """Create sample alert data for testing."""
    return {
        'interface_name': 'SG-CORE-01-GE-0/0/1',
        'node_name': 'TH1VCGH1_70',
        'detection_time': '2024-12-19 14:30:25',
        'anomaly_type': 'Traffic Surge',
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
        'traffic_chart_placeholder': '📈 Real-time traffic chart showing 185% threshold breach',
        'performance_chart_placeholder': '📊 Performance degradation metrics and KPI trends',
        'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'system_id': 'CORE-AI-001'
    }


if __name__ == "__main__":
    """Demo the email service with sample data."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create email service
    email_service = EmailService()
    
    # Create sample alert data
    sample_data = create_sample_alert_data()
    
    print("🔧 Testing Email Service...")
    print(f"📧 SMTP Host: {email_service.smtp_host}:{email_service.smtp_port}")
    print(f"👤 Username: {email_service.username}")
    
    # Test connection
    print("\n🌐 Testing SMTP connection...")
    if email_service.test_connection():
        print("✅ SMTP connection successful!")
    else:
        print("❌ SMTP connection failed!")
    
    # Create email (but don't send in demo)
    print("\n📝 Creating sample email...")
    try:
        message = email_service.create_sg_rg_alert_email(
            alert_data=sample_data,
            recipients=['test@example.com']  # Demo recipient
        )
        
        print("✅ Email message created successfully!")
        print(f"📋 Subject: {message['Subject']}")
        print(f"📬 Recipients: {message['To']}")
        
        # Save HTML version for preview
        html_content = email_service.format_sg_rg_alert_template(sample_data)
        preview_path = Path('temp') / 'email_preview.html'
        preview_path.parent.mkdir(exist_ok=True)
        
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 Email preview saved to: {preview_path}")
        print("🌐 Open the preview file in your browser to see the template!")
        
    except Exception as e:
        print(f"❌ Failed to create email: {e}") 