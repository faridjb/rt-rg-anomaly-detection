# Network KPI Anomaly Detection & Alerting System

An automated system for monitoring network Key Performance Indicators (KPIs), detecting anomalies using Prophet AI, generating insightful visualizations, and sending email alerts with embedded charts.

## 🚀 Quick Start

```bash
# 1. Set up environment
source setup_env.sh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test with demo workflow
python scripts/demo_anomaly_workflow.py --test

# 4. Run production workflow
python scripts/demo_anomaly_workflow.py
```

## ✅ Features

- **🤖 AI Anomaly Detection**: Uses Facebook Prophet to identify unusual patterns in network traffic data
- **📧 Email-Compatible Alerts**: Generates HTML email alerts with embedded charts for maximum compatibility
- **📊 Dynamic Visualizations**: Creates plots with auto-adjusted time axis labels for optimal readability
- **🗄️ Oracle Database Integration**: Fetches KPI data from Oracle database (10.200.6.227:1521)
- **⚙️ Environment-Based Configuration**: Settings managed via `.env` file and environment variables
- **🔄 Production Ready**: Comprehensive error handling, logging, and workflow orchestration
- **⏰ Cron Scheduling**: Includes setup for automated execution

## 📋 Prerequisites

### System Requirements
- Python 3.7+
- Oracle Instant Client (for database connectivity)
- Network access to Oracle DB and email server

### Database Configuration
- **Oracle Database**: 10.200.6.227:1521 (fcsouth.rightel.ir)
- **User**: tools_ml
- **Required Tables**: FOCUSADM.H3G_CG_RATING_MAINTABLE

### Email Configuration  
- **SMTP Server**: mail.rightel.ir:465
- **User**: Performance-Dev
- **Security**: TLS/SSL enabled

## 🔧 Installation & Setup

### 1. Environment Setup
```bash
# Clone/navigate to project directory
cd core_rg

# Set up environment variables
source setup_env.sh

# Copy and configure environment file
cp env_template.txt .env
# Edit .env with your credentials
nano .env
```

### 2. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Core packages for anomaly detection
pip install prophet pandas numpy matplotlib seaborn cx_Oracle python-dotenv
```

### 3. Oracle Client Setup (if needed)
```bash
# For Ubuntu/Debian
sudo apt-get install oracle-instantclient19.3-basic

# Set environment variables (already in setup_env.sh)
export LD_LIBRARY_PATH=/usr/lib/oracle/19.3/client64/lib:$LD_LIBRARY_PATH
export ORACLE_HOME=/usr/lib/oracle/19.3/client64
```

## 🎯 Usage

### Demo Mode (Recommended for testing)
```bash
# Test with simulated data (no database required)
python scripts/demo_anomaly_workflow.py --test

# Analyze specific node
python scripts/demo_anomaly_workflow.py --node TH1VCGH1_70 --test

# Limit number of nodes processed
python scripts/demo_anomaly_workflow.py --max-nodes 5
```

### Production Mode
```bash
# Run with real Oracle database
python scripts/demo_anomaly_workflow.py

# Analyze specific KPI
python scripts/demo_anomaly_workflow.py --kpi rg_downlink

# Send to test recipient
python scripts/demo_anomaly_workflow.py --test
```

### Command Line Options
```bash
--test              # Send emails to test recipient instead of production
--node NODE_NAME    # Analyze specific node only
--kpi KPI_COLUMN    # Analyze specific KPI (default: total_rg)
--max-nodes N       # Limit number of nodes processed
```

## 🏗️ System Architecture

### Project Structure
```
core_rg/
├── src/                          # Main source code
│   ├── utils/                    # Configuration and logging utilities
│   ├── data/                     # Data models and database layer
│   ├── services/                 # Business logic services
│   └── core/                     # Core functionality
├── scripts/
│   ├── demo_anomaly_workflow.py  # ✅ Main production workflow
│   ├── run_monitoring.py         # ✅ System orchestrator
│   └── test_email_template.py    # ✅ Email testing utility
├── config/                       # Configuration files
├── templates/                    # Email and chart templates
│   └── email_templates/
│       └── sg_rg_traffic_alert.html
├── output/                       # Generated outputs
│   └── charts/                   # Chart files
├── logs/                         # Log files
├── setup_env.sh                 # ✅ Environment setup script
├── env_template.txt             # ✅ Environment template
├── requirements.txt             # ✅ Python dependencies
└── pyproject.toml              # ✅ Project configuration
```

### Core Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Main Workflow** | Production anomaly detection workflow | `scripts/demo_anomaly_workflow.py` |
| **Database Service** | Oracle DB connectivity and KPI data fetching | `src/data/database.py` |
| **Anomaly Detector** | Prophet-based anomaly detection engine | `src/services/` |
| **Visualization** | Chart generation with dynamic formatting | `src/services/` |
| **Email Service** | HTML email composition and sending | `src/services/` |
| **Configuration** | Environment-based configuration management | `src/utils/` |

## 📊 Workflow Implementation

The system implements this complete monitoring workflow:

### 1. Database Query & Node Discovery
```sql
-- Get distinct active nodes
SELECT DISTINCT SNODE 
FROM FOCUSADM.H3G_CG_RATING_MAINTABLE 
WHERE SDATE >= SYSDATE - 24/24
ORDER BY SNODE
```

### 2. Historical Data Retrieval (per node)
```sql
-- Get 30-day historical KPI data
SELECT SDATE,
       CNT1_167774004 AS rg_downlink,
       CNT2_167774004 AS rg_uplink,
       (CNT1_167774004 + CNT2_167774004) AS total_rg
FROM FOCUSADM.H3G_CG_RATING_MAINTABLE
WHERE SNODE = :node_name 
AND SDATE >= SYSDATE - 30
ORDER BY SDATE
```

### 3. Prophet Anomaly Detection
- Trains Prophet model on 30-day historical data
- Detects anomalies using statistical bounds
- Checks last 4 hours for recent anomalies
- Configurable sensitivity (0.8-0.99 range)

### 4. Visualization & Alerting
- Generates KPI plots with confidence bounds
- Highlights anomaly points in red
- Creates 36-hour and 30-day analysis views
- Sends HTML email alerts with embedded charts

### 5. Email Notifications
- Uses email-compatible HTML template
- Embeds high-resolution charts (300 DPI)
- Includes anomaly summary and statistics
- Supports test and production recipients

## 📧 Email Template System

### Template Location
```
templates/email_templates/sg_rg_traffic_alert.html
```

### Template Features
- **📱 Email Client Compatibility**: Table-based layout with inline CSS
- **📊 Embedded Charts**: Base64-encoded images for universal support  
- **📋 Anomaly Summary**: Detailed detection results table
- **🎨 Professional Design**: Modern, responsive HTML structure
- **🚨 Severity Indicators**: Color-coded alert levels

### Email Recipients
- **Production**: EX.F.Jabarimaleki@rightel.ir
- **Test Mode**: Configurable via EMAIL_TEST_RECIPIENT in .env

## ⏰ Automated Scheduling

### Cron Setup (Linux/macOS)
```bash
# Edit crontab
crontab -e

# Run every 4 hours
0 */4 * * * cd /home/tools/Documents/core_rg && /home/tools/Documents/core_rg/.venv/bin/python /home/tools/Documents/core_rg/scripts/demo_anomaly_workflow.py >> /home/tools/Documents/core_rg/cron_output.log 2>&1
```

### Cron Best Practices
- Use absolute paths for Python interpreter and script
- Change to project directory before execution
- Redirect output to log file for debugging
- Test manually before scheduling

## 🔍 Monitoring & Debugging

### Log Files
```bash
# Application logs
tail -f logs/app.log

# Cron execution logs  
tail -f cron_output.log

# Check log level in .env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Health Checks
```bash
# Test database connectivity
python -c "from src.data.database import DatabaseService; db=DatabaseService(); print('DB OK' if db.test_connection() else 'DB FAIL')"

# Test email server
python -c "import smtplib; server=smtplib.SMTP_SSL('mail.rightel.ir', 465); print('Email OK')"

# Test configuration
python -c "from src.utils.config import get_config; print('Config OK')"
```

### Common Troubleshooting
```bash
# Prophet installation issues
conda install -c conda-forge prophet

# Oracle client issues  
sudo apt-get install oracle-instantclient19.3-basic

# Permission errors
chmod +x setup_env.sh
chmod -R 755 logs/ output/ temp/
```

## 📈 KPI Metrics Monitored

### Primary KPIs
- **Total RG Traffic**: Combined uplink + downlink traffic
- **RG Downlink**: Downlink traffic volume  
- **RG Uplink**: Uplink traffic volume

### Detection Thresholds
- **Sensitivity**: 95% confidence interval (configurable)
- **Time Window**: Last 4 hours for recent anomaly detection
- **Training Period**: 30 days historical data
- **Alert Threshold**: 2+ anomalies in monitoring window

## 🔒 Security & Configuration

### Environment Variables (.env)
```bash
# Database credentials
DATABASE_HOST=10.200.6.227
DATABASE_PORT=1521  
DATABASE_USER=tools_ml
DATABASE_PASSWORD=your_password
DATABASE_SERVICE=fcsouth.rightel.ir

# Email configuration
EMAIL_SMTP_HOST=mail.rightel.ir
EMAIL_SMTP_PORT=465
EMAIL_USERNAME=Performance-Dev
EMAIL_PASSWORD=your_password
EMAIL_FROM_ADDRESS=Performance-Dev@rightel.ir
EMAIL_TO=EX.F.Jabarimaleki@rightel.ir
EMAIL_TEST_RECIPIENT=your-test@rightel.ir

# System settings
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key
```

### Security Notes
⚠️ **Important**: 
- Store credentials securely
- Use environment-specific .env files
- Consider secrets management for production
- Regularly rotate passwords and keys

## 🚀 Future Enhancements

- **🎫 Ticketing Integration**: Automated trouble ticket creation
- **🌐 Web Dashboard**: Real-time monitoring interface  
- **📱 Mobile Alerts**: SMS and push notifications
- **🔄 Multi-Model Detection**: Ensemble anomaly detection
- **☁️ Cloud Deployment**: Containerized deployment options
- **📊 Advanced Analytics**: Trend analysis and forecasting

## 📞 Support

**Contact Information:**
- **Email**: Performance-Tools@rightel.ir
- **System**: Network Performance Monitoring Team

**For Issues:**
1. Check logs in `./logs/` directory
2. Verify environment setup with `source setup_env.sh`  
3. Test connectivity and configuration
4. Review cron_output.log for scheduling issues

---

**🎯 Complete Network KPI Anomaly Detection System - Ready for Production!** 🚀
