@echo off
echo ===================================================
echo Launching KASMU v4 Workbench ^& Dashboard
echo ===================================================

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Starting API Server...
python Simulation/api_server.py

pause
