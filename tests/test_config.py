#!/usr/bin/env python3
"""
Simple test script to verify configuration and environment setup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_simple import get_config

def test_config():
    """Test configuration loading and environment variables."""
    print("=" * 60)
    print("Network Monitoring System - Configuration Test")
    print("=" * 60)
    
    try:
        # Load configuration
        config = get_config()
        
        # Test database configuration
        print("\n📊 Database Configuration:")
        db_config = config.database
        print(f"  Username: {db_config.username}")
        print(f"  Host: {db_config.host}:{db_config.port}")
        print(f"  DSN: {db_config.dsn}")
        print(f"  Password: {'✓ Set' if db_config.password else '✗ Missing'}")
        
        # Test ticketing configuration
        print("\n🎫 Ticketing System Configuration:")
        ticket_config = config.ticketing
        print(f"  API URL: {ticket_config.api_url}")
        print(f"  Username: {ticket_config.username}")
        print(f"  Password: {'✓ Set' if ticket_config.password else '✗ Missing'}")
        print(f"  Bearer Token: {'✓ Set' if ticket_config.bearer_token else '✗ Missing'}")
        
        # Test email configuration
        print("\n📧 Email Configuration:")
        email_config = config.email
        print(f"  SMTP Server: {email_config.smtp_server}:{email_config.smtp_port}")
        print(f"  Username: {email_config.username}")
        print(f"  From Address: {email_config.from_address}")
        print(f"  Password: {'✓ Set' if email_config.password else '✗ Missing'}")
        print(f"  Use TLS: {email_config.use_tls}")
        
        # Test email recipients
        if email_config.recipients.get('to'):
            print(f"  To: {', '.join(email_config.recipients['to'])}")
        if email_config.recipients.get('cc'):
            print(f"  CC: {', '.join(email_config.recipients['cc'])}")
        
        # Test KPI configuration
        print("\n🔍 KPI Monitoring Configuration:")
        enabled_kpis = config.get_enabled_kpis()
        for kpi in enabled_kpis:
            print(f"  ✓ {kpi.name} ({kpi.detection_algorithm})")
        
        # Test file paths
        print("\n📁 File Configuration:")
        print(f"  Charts: ./output/charts")
        print(f"  Logs: ./logs") 
        print(f"  Templates: ./templates")
        
        # Create directories
        config.create_directories()
        print(f"  ✓ Directories created")
        
        print("\n" + "=" * 60)
        print("✅ Configuration test completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test that environment variables are set correctly."""
    print("\n🔧 Environment Variables Check:")
    
    required_vars = [
        'DB_USERNAME', 'DB_PASSWORD', 'DB_HOST', 'DB_DSN',
        'TICKETING_API_URL', 'TICKETING_USERNAME', 'TICKETING_PASSWORD', 'TICKETING_BEARER_TOKEN',
        'EMAIL_SMTP_SERVER', 'EMAIL_USERNAME', 'EMAIL_PASSWORD', 'EMAIL_FROM_ADDRESS'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: Set")
        else:
            print(f"  ✗ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please run: source setup_env.sh")
        return False
    else:
        print("\n✅ All required environment variables are set!")
        return True

if __name__ == "__main__":
    print("Starting configuration and environment test...")
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test configuration
    config_ok = test_config()
    
    if env_ok and config_ok:
        print("\n🎉 All tests passed! System is ready for use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        sys.exit(1) 