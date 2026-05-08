@echo off
REM Setup script for Vision ROI Reconstructor - Task #01

echo ===============================================
echo Vision ROI Reconstructor - Task #01 Setup
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Downloading YOLO11 model...
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo.
echo ===============================================
echo Setup Complete!
echo ===============================================
echo.
echo Next steps:
echo 1. Place your video files in the ./data folder
echo 2. Run: python task_01_roi_reconstructor.py
echo 3. Check ./output folder for processed videos
echo.
echo For more information, see TASK_01_README.md
echo.
pause
