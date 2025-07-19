#!/usr/bin/env python3
"""
Test script to preview the new beautiful email template
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

# Import from main workflow
from main_anomaly_workflow import EmailNotificationService

def test_email_template():
    """Test the new email template design."""
    
    print("🎨 Testing New Beautiful Email Template")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'total_rg': [15422.0, 16800.0, 14900.0],  # Sample traffic data
        'timestamp': pd.date_range('2025-05-23', periods=3, freq='H')
    })
    
    # Create sample detection result
    detection_result = {
        'success': True,
        'data': sample_data,
        'total_anomalies': 24,
        'anomaly_rate': 0.034,
        'recent_check': {
            'recent_anomaly_count': 0,
            'severity': 'LOW'
        }
    }
    
    # Create email service and generate template
    email_service = EmailNotificationService()
    html_content = email_service._create_email_body("ES1CGH1_102", detection_result)
    
    # Save preview to file
    preview_path = Path('temp/email_preview_new_template.html')
    preview_path.parent.mkdir(exist_ok=True)
    
    with open(preview_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ New email template preview saved to: {preview_path}")
    print("\n🎉 Template Features:")
    print("   • Modern gradient backgrounds")
    print("   • Professional card-based layout") 
    print("   • Grid stats display")
    print("   • Beautiful color schemes")
    print("   • Enhanced typography")
    print("   • Professional table styling")
    print("   • Recommendations section")
    print("   • Modern header and footer")
    
    print(f"\n📧 Template ready for email notifications!")
    print(f"   Preview file: {preview_path.absolute()}")

if __name__ == "__main__":
    test_email_template() 