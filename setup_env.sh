#!/bin/bash
# Environment setup script for Network Monitoring System
# Run this script to set environment variables: source setup_env.sh

echo "Setting up environment variables for Network Monitoring System..."

# Database Configuration (from config_core.py) - Fixed variable names
export DATABASE_USER="tools_ml"
export DATABASE_PASSWORD="Focu\$2021"
export DATABASE_SERVICE="fcsouth.rightel.ir"
export DATABASE_PORT="1521"
export DATABASE_HOST="10.200.6.227"
export DB_ENCODING="UTF-8"

# Ticketing System Configuration (from raise_ticket.py)
export TICKETING_API_URL="https://10.201.6.13/KM_UCMS_TT"
export TICKETING_USERNAME="EX.F.Jabarimaleki"
export TICKETING_PASSWORD="7s&aBvP/#49.qjC7"
export TICKETING_BEARER_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJLTV9VQ01TIiwiaWF0IjoxNjQxMjAwOTQ3LCJleHAiOjIzMDM4ODg5NDcsImF1ZCI6Ind3dy5rbS1lbC5pciIsInN1YiI6ImluZm9Aa20tZWwuaXIifQ.3e3MuRM_TktHSwi7Vitbj5N0zR0E1Dg2H_t1T1zE0ZY"

# Email Configuration (from send_ticket_core.py)
export EMAIL_SMTP_HOST="mail.rightel.ir"
export EMAIL_SMTP_PORT="465"
export EMAIL_USERNAME="Performance-Dev"
export EMAIL_PASSWORD="Focus2021"
export EMAIL_FROM_ADDRESS="Performance-Dev@rightel.ir"
export EMAIL_USE_TLS="true"

# Email Recipients
export EMAIL_TO="Performance-Core@rightel.ir"
export EMAIL_CC="Performance-Tools@rightel.ir,EX.F.Jabarimaleki@rightel.ir"

# Application Configuration
export LOG_LEVEL="INFO"
export DEBUG_MODE="false"
export ENABLE_METRICS="false"

# Security Configuration (CHANGE THESE!)
export SECRET_KEY="your_secret_key_here_change_this"
export ENCRYPTION_KEY="your_encryption_key_here_change_this"

echo "Environment variables set successfully!"
echo ""
echo "Note: Please change the SECRET_KEY and ENCRYPTION_KEY values for security."
echo "You can verify the setup by running: python scripts/run_monitoring.py --health-check" 