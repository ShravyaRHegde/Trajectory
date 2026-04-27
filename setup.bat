@echo off
echo ===================================================
echo KASMU v4 Environment Setup
echo ===================================================

echo [1/3] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Ensure Python is installed and in PATH.
    pause
    exit /b %errorlevel%
)

echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo [3/3] Setup complete! You can now use run.bat to start the simulation.
pause
