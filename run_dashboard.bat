@echo off
title MQL5 Trading Analytics Dashboard
echo.
echo ========================================
echo   MQL5 TRADING ANALYTICS DASHBOARD
echo   Professional Wall Street Analysis
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update requirements
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [INFO] Starting dashboard...
echo [INFO] Open your browser at http://localhost:8501
echo [INFO] Press Ctrl+C to stop the server
echo.

streamlit run app.py --server.headless true

pause
