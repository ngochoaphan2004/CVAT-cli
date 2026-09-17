@echo off
echo =======================================
echo CVAT-CLI Environment Setup
echo =======================================

echo 1. Creating Python virtual environment (.venv)...
python -m venv .venv

echo 2. Upgrading pip...
.\.venv\Scripts\python.exe -m pip install --upgrade pip

echo 3. Installing PyTorch with CUDA 12.4 support...
.\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo 4. Installing CVAT SDK, Transformers, and other dependencies...
.\.venv\Scripts\pip.exe install cvat-sdk transformers opencv-python numpy Pillow rasterio shapely topojson

echo =======================================
echo Setup Complete!
echo You can now run the Python scripts inside .venv
echo =======================================
pause

