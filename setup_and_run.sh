#!/bin/bash
# PM2.5 Estimation System - Quick Setup Script for Linux/Mac
# Run this script to set up and launch the application

echo "============================================================"
echo "PM2.5 ESTIMATION SYSTEM - QUICK SETUP"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

echo "[1/5] Python detected successfully"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
else
    echo "[2/5] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "[4/5] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo ""

# Launch application
echo "[5/5] Launching PM2.5 Estimation System..."
echo ""
echo "============================================================"
echo "Server will start at: http://127.0.0.1:5000"
echo "Press CTRL+C to stop the server"
echo "============================================================"
echo ""

python app.py
