# Fusion AI - Installation Guide

Complete installation instructions for setting up Fusion AI with virtual environment isolation.

## Prerequisites

- **Python 3.10+** (Python 3.8-3.9 are no longer supported)
- **CUDA-capable GPU** (NVIDIA, recommended for performance)
- **8GB+ VRAM** (16GB+ recommended for larger models)
- **Blackmagic Design Fusion 9+** or **DaVinci Resolve 16+**
- **Git** (for cloning repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/fusion_ai.git
cd fusion_ai
```

### 2. Create Python Virtual Environment

**Using venv (recommended):**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Alternative - Using conda:**

```bash
conda create -n fusion_ai python=3.10
conda activate fusion_ai
```

### 3. Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install fusion_ai package in editable mode
pip install -e .
```

This will install all core dependencies from `pyproject.toml`:
- PyTorch (with CUDA support)
- Transformers
- Diffusers
- Pillow, NumPy, OpenCV
- And more...

### 4. Install Optional Dependencies

**For all AI features (recommended):**
```bash
pip install -e ".[all]"
```

**Or install specific feature sets:**

```bash
# Inpainting (LaMa, Stable Diffusion)
pip install -e ".[inpainting]"

# Upscaling (Real-ESRGAN)
pip install -e ".[upscaling]"

# ControlNet support
pip install controlnet-aux

# MediaPipe (OpenPose fallback)
pip install mediapipe
```

### 5. Verify Installation

```bash
# Check if fusion_ai is importable
python -c "import fusion_ai; print(fusion_ai.__version__)"

# List installed models
python -c "from fusion_ai.models import *; print('All models imported successfully')"
```

### 6. Install Fusion Fuses

**Automatic installation (recommended):**

```bash
# For Fusion Studio only (default)
python scripts/configure_fuses.py

# For DaVinci Resolve only
python scripts/configure_fuses.py --resolve

# For BOTH Fusion Studio and DaVinci Resolve
python scripts/configure_fuses.py --both
```

The script will:
- Auto-detect the correct Fuses directory
- Copy all .fuse files
- Configure each fuse with your venv Python path
- Work on Windows/macOS/Linux

**Manual installation:**

Copy `.fuse` files from `fusion_ai/fuses/` to the appropriate directory:

**Fusion Studio:**
- Windows: `%APPDATA%\Blackmagic Design\Fusion\Fuses\`
- macOS: `~/Library/Application Support/Blackmagic Design/Fusion/Fuses/`
- Linux: `~/.fusion/BlackmagicDesign/Fusion/Fuses/`

**DaVinci Resolve:**
- Windows: `%APPDATA%\Blackmagic Design\DaVinci Resolve\Fusion\Fuses\`
- macOS: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses/`
- Linux: `~/.local/share/DaVinci Resolve/Fusion/Fuses/`

**13 AI Fuse files:**
- `DepthAnythingV2.fuse` - Depth estimation
- `QwenEdit.fuse` - Vision-language editing
- `AIInpaint.fuse` - Object removal (basic)
- `AITemporalInpaint.fuse` - Object removal (temporal)
- `AIWANInpaint.fuse` - Object removal (advanced WAN)
- `AIOutpaint.fuse` - Image extension
- `AITemporalOutpaint.fuse` - Image extension (temporal)
- `AIUpscale.fuse` - AI super-resolution
- `AIStyleTransfer.fuse` - Style transfer
- `AIFrameExtend.fuse` - Frame generation
- `AIVideoGenerate.fuse` - WAN video generation
- `AIControlNet.fuse` - ControlNet (pose/edge/depth)

**Note:** If manual install, you must edit each .fuse file to set the Python path. See [docs/INSTALLATION_LOCATIONS.md](docs/INSTALLATION_LOCATIONS.md).

### 7. Configure Model Cache Directory (Optional)

Set environment variable to control where models are downloaded:

```bash
# Linux/macOS - add to ~/.bashrc or ~/.zshrc
export FUSION_AI_CACHE_DIR="$HOME/.cache/fusion_ai"
export FUSION_AI_DEVICE="cuda"  # or "cpu"

# Windows - set in System Environment Variables
setx FUSION_AI_CACHE_DIR "%USERPROFILE%\.cache\fusion_ai"
setx FUSION_AI_DEVICE "cuda"
```

## Platform-Specific Notes

### Windows

1. Make sure Visual Studio C++ Build Tools are installed for compiling some dependencies
2. If you encounter CUDA errors, ensure NVIDIA drivers are up to date
3. Use PowerShell or Command Prompt as Administrator for installation

### macOS

1. For M1/M2 Macs, PyTorch will use MPS (Metal Performance Shaders)
2. Some dependencies may require Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

### Linux

1. Ensure CUDA toolkit matches your PyTorch version
2. May need to install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev python3-venv libgl1-mesa-glx

   # Fedora/RHEL
   sudo dnf install python3-devel mesa-libGL
   ```

## Verifying GPU Support

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output (if GPU available):
```
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 3090
```

## Troubleshooting

### Issue: "No module named 'fusion_ai'"

**Solution:** Make sure you activated the venv and installed the package:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

### Issue: CUDA out of memory

**Solutions:**
1. Use FP16 precision instead of FP32 (check Fusion node inspector)
2. Reduce image resolution
3. Close other GPU-intensive applications
4. Use smaller model variants (e.g., DepthAnything "small" instead of "large")

### Issue: Models downloading slowly

**Solution:** Models are cached after first download. First run may take time depending on internet speed.

### Issue: Import errors for optional dependencies

**Solution:** Install the specific feature set:
```bash
pip install -e ".[all]"
```

### Issue: Fusion doesn't see the fuses

**Solutions:**
1. Verify fuses are in correct directory
2. Restart Fusion/Resolve
3. Check file permissions (should be readable)
4. Ensure Python path in fuses points to your venv Python:
   ```lua
   -- In .fuse files, Python is called via system path
   -- Make sure venv is activated when Fusion runs
   ```

## Updating

To update to the latest version:

```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Pull latest changes
git pull origin main

# Update dependencies
pip install -e ".[all]" --upgrade

# Reinstall fuses (if updated)
python scripts/install_fuses.py
```

## Uninstallation

```bash
# Deactivate venv
deactivate

# Remove virtual environment
rm -rf venv  # Linux/macOS
# or
rmdir /s venv  # Windows

# Remove Fusion fuses manually from Fuses directory

# Remove model cache (optional)
rm -rf ~/.cache/fusion_ai  # Linux/macOS
# or
rmdir /s %USERPROFILE%\.cache\fusion_ai  # Windows
```

## Development Installation

For developers who want to contribute:

```bash
# Clone with development branch
git clone -b develop https://github.com/your-org/fusion_ai.git
cd fusion_ai

# Create venv
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Next Steps

After installation, see:
- **[README.md](README.md)** - Feature overview and quick start
- **[USAGE.md](USAGE.md)** - Detailed usage examples
- **[examples/](examples/)** - Example workflows
- **[docs/](docs/)** - Complete documentation

## Support

For installation issues:
1. Check troubleshooting section above
2. Search existing GitHub Issues
3. Create new issue with:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce
