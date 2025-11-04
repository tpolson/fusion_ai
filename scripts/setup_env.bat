@echo off
REM Setup script for Fusion AI package (Windows)

echo Setting up Fusion AI environment...

REM Check Python version
python --version

REM Check for CUDA
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo CUDA detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo CUDA not detected, will use CPU inference
)

REM Create virtual environment (optional)
set /p VENV="Create virtual environment? (y/n) "
if /i "%VENV%"=="y" (
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated
)

REM Install package
echo Installing fusion-ai package...
pip install -e .

REM Install development dependencies
set /p DEV="Install development dependencies? (y/n) "
if /i "%DEV%"=="y" (
    pip install -e .[dev]
)

REM Install Fuse plugins
echo.
set /p FUSES="Install Fusion Fuse plugins? (y/n) "
if /i "%FUSES%"=="y" (
    python scripts\install_fuses.py
)

echo.
echo Setup complete!
echo.
echo To use the package:
echo   from fusion_ai.models import DepthAnythingV2, QwenEdit
echo.
echo To test installation:
echo   python -c "from fusion_ai import DepthAnythingV2; print('Success!')"

pause
