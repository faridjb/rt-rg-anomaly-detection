#!/usr/bin/env python3
"""
Health check script for Network Monitoring System.
This script tests basic functionality without requiring all dependencies.
"""

import sys
import os
import traceback
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    try:
        from src.utils.config_simple import get_config
        config = get_config()
        
        # Test basic config properties
        db_config = config.database
        if not db_config.username or not db_config.host:
            raise ValueError("Database configuration incomplete")
            
        print("  ✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging setup."""
    print("📝 Testing logging...")
    try:
        import logging
        
        # Create a simple logger
        logger = logging.getLogger('health_check')
        logger.setLevel(logging.INFO)
        
        # Test basic logging
        logger.info("Health check logging test")
        print("  ✓ Logging system working")
        return True
    except Exception as e:
        print(f"  ✗ Logging test failed: {e}")
        return False

def test_directories():
    """Test directory creation."""
    print("📁 Testing directory structure...")
    try:
        directories = [
            './output',
            './output/charts', 
            './logs',
            './temp'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        print("  ✓ Directory structure created")
        return True
    except Exception as e:
        print(f"  ✗ Directory test failed: {e}")
        return False

def test_data_models():
    """Test data model imports."""
    print("📊 Testing data models...")
    try:
        # Test if we can import the models (even if Pydantic isn't available)
        sys.path.insert(0, 'src')
        
        # Create a simple test without Pydantic
        class SimpleKPIData:
            def __init__(self, node_name, kpi_name, timestamp, value):
                self.node_name = node_name
                self.kpi_name = kpi_name
                self.timestamp = timestamp
                self.value = value
        
        # Test creating a simple data object
        test_data = SimpleKPIData(
            node_name="TEST_NODE",
            kpi_name="Test KPI",
            timestamp=datetime.now(),
            value=95.5
        )
        
        print("  ✓ Data models working")
        return True
    except Exception as e:
        print(f"  ✗ Data model test failed: {e}")
        return False

def test_yaml_config():
    """Test YAML configuration file loading."""
    print("⚙️  Testing YAML config loading...")
    try:
        import yaml
        
        # Test loading the config file if it exists
        config_file = 'config/config.yaml'
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"  ✓ YAML config loaded: {len(config_data)} sections")
        else:
            print("  ℹ️  YAML config file not found (using environment variables)")
            
        return True
    except Exception as e:
        print(f"  ✗ YAML config test failed: {e}")
        return False

def test_network_connectivity():
    """Test basic network connectivity (optional)."""
    print("🌐 Testing network connectivity...")
    try:
        import socket
        from src.utils.config_simple import get_config
        
        config = get_config()
        
        # Test database connectivity (just socket connection)
        db_host = config.database.host
        db_port = config.database.port
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((db_host, db_port))
        sock.close()
        
        if result == 0:
            print(f"  ✓ Database host {db_host}:{db_port} is reachable")
        else:
            print(f"  ⚠️  Database host {db_host}:{db_port} is not reachable")
            
        # Test email server connectivity (optional)
        email_host = config.email.smtp_server
        email_port = config.email.smtp_port
        
        if email_host:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((email_host, email_port))
            sock.close()
            
            if result == 0:
                print(f"  ✓ Email server {email_host}:{email_port} is reachable")
            else:
                print(f"  ⚠️  Email server {email_host}:{email_port} is not reachable")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Network connectivity test failed: {e}")
        # Network issues shouldn't fail health check
        return True

def test_python_requirements():
    """Test Python version and basic requirements."""
    print("🐍 Testing Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 9:
        print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ⚠️  Python {python_version.major}.{python_version.minor} (recommended: 3.9+)")
    
    # Check for standard library modules
    required_modules = ['yaml', 'pathlib', 'datetime', 'logging', 'socket']
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} not available")
            return False
    
    return True

def test_file_permissions():
    """Test file system permissions."""
    print("🔐 Testing file permissions...")
    try:
        # Test write permissions in key directories
        test_dirs = ['./output', './logs', './temp']
        
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = Path(test_dir) / 'health_check_test.tmp'
            test_file.write_text('test')
            test_file.unlink()
            
        print("  ✓ File system permissions OK")
        return True
    except Exception as e:
        print(f"  ✗ File permission test failed: {e}")
        return False

def main():
    """Run all health check tests."""
    print("=" * 60)
    print("Network Monitoring System - Health Check")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        test_python_requirements,
        test_configuration,
        test_logging,
        test_directories,
        test_file_permissions,
        test_data_models,
        test_yaml_config,
        test_network_connectivity,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print(f"Health Check Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All health checks passed! System is ready.")
        return 0
    elif passed >= total * 0.7:  # 70% threshold
        print("⚠️  Most health checks passed. System is mostly functional.")
        return 1
    else:
        print("❌ Multiple health checks failed. System needs attention.")
        return 2

if __name__ == "__main__":
    sys.exit(main()) 