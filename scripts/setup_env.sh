#!/bin/bash
# Setup script for Fusion AI package

set -e

echo "Setting up Fusion AI environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "CUDA not detected, will use CPU inference"
fi

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated"
fi

# Install package
echo "Installing fusion-ai package..."
pip install -e .

# Install development dependencies
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]"
fi

# Install Fuse plugins
echo ""
read -p "Install Fusion Fuse plugins? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/install_fuses.py
fi

echo ""
echo "Setup complete!"
echo ""
echo "To use the package:"
echo "  from fusion_ai.models import DepthAnythingV2, QwenEdit"
echo ""
echo "To test installation:"
echo "  python -c 'from fusion_ai import DepthAnythingV2; print(\"Success!\")'"
