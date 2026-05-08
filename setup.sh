#!/bin/bash

# Setup script for Vision ROI Reconstructor - Task #01

echo "==============================================="
echo "Vision ROI Reconstructor - Task #01 Setup"
echo "==============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[4/4] Downloading YOLO11 model..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Place your video files in the ./data folder"
echo "2. Run: python task_01_roi_reconstructor.py"
echo "3. Check ./output folder for processed videos"
echo ""
echo "For more information, see TASK_01_README.md"
echo ""
