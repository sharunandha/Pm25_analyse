@echo off
REM PM2.5 Estimation System - Quick Setup Script for Windows
REM Run this script to set up and launch the application

echo ============================================================
echo PM2.5 ESTIMATION SYSTEM - QUICK SETUP
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo [1/5] Python detected successfully
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
) else (
    echo [2/5] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo [4/5] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
echo.

REM Launch application
echo [5/5] Launching PM2.5 Estimation System...
echo.
echo ============================================================
echo Server will start at: http://127.0.0.1:5000
echo Press CTRL+C to stop the server
echo ============================================================
echo.

python app.py

pause
