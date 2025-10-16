@echo off
echo ========================================
echo Productivity Tracker - Headless Mode
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if config file exists
if not exist "config.yaml" (
    echo ERROR: config.yaml not found
    echo Please ensure the configuration file exists
    pause
    exit /b 1
)

echo Starting headless productivity tracker...
echo.
echo The system will:
echo - Run 24/7 in the background
echo - Capture data daily from 7am to 5pm
echo - Store data in PostgreSQL database
echo - Log all activities to productivity_tracker.log
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start the headless tracker
python start_headless.py

echo.
echo System stopped.
pause
