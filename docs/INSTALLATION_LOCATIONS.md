# Fusion AI Installation Locations

## Directory Structure Overview

The fusion_ai package and Fusion are installed in **separate locations**:

```
System Layout:
├── /path/to/fusion_ai/           # Can be anywhere (your project folder)
│   ├── venv/                      # Python virtual environment
│   ├── fusion_ai/                 # Python package
│   └── ...
│
└── Fusion Installation/
    └── Fuses/                     # Only .fuse files go here
        ├── DepthAnythingV2.fuse
        ├── AIControlNet.fuse
        └── ...
```

## Installation Locations by Platform

### Python Package (fusion_ai)
Can be installed **anywhere** - typically in your projects folder:

```bash
# Example locations:
~/projects/fusion_ai/              # Linux/macOS
C:\Users\YourName\fusion_ai\       # Windows
/opt/fusion_ai/                    # Linux (system-wide)
```

### Fusion Fuses (.fuse files)
Must be in **Fusion's Fuses directory**:

**Windows:**
```
%APPDATA%\Blackmagic Design\Fusion\Fuses\
# Typical: C:\Users\YourName\AppData\Roaming\Blackmagic Design\Fusion\Fuses\
```

**macOS:**
```
~/Library/Application Support/Blackmagic Design/Fusion/Fuses/
```

**Linux:**
```
~/.fusion/BlackmagicDesign/Fusion/Fuses/
# OR
~/.local/share/Blackmagic Design/Fusion/Fuses/
```

## How Fusion Finds Python

The **critical part**: Lua fuses call Python via system commands. There are three approaches:

### Option 1: Activate venv Globally (Recommended for Development)

```bash
# Activate venv before starting Fusion
source /path/to/fusion_ai/venv/bin/activate
# Now launch Fusion
/path/to/fusion/fusion
```

When venv is activated:
- `python` command points to venv's Python
- fusion_ai package is importable
- All dependencies available

### Option 2: Configure Python Path in Fuses (Recommended for Production)

Edit each .fuse file to use absolute path to venv Python:

```lua
-- Before (uses system Python)
local python_cmd = "python -c ..."

-- After (uses venv Python)
local python_cmd = "/path/to/fusion_ai/venv/bin/python -c ..."
```

**Example for Windows:**
```lua
local python_cmd = "C:/Users/YourName/fusion_ai/venv/Scripts/python.exe -c ..."
```

### Option 3: Set Environment Variables

Set `FUSION_AI_PYTHON` environment variable:

**Linux/macOS** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export FUSION_AI_PYTHON="/path/to/fusion_ai/venv/bin/python"
```

**Windows** (System Properties → Environment Variables):
```
Variable: FUSION_AI_PYTHON
Value: C:\Users\YourName\fusion_ai\venv\Scripts\python.exe
```

Then modify fuses to use it:
```lua
local python_path = os.getenv("FUSION_AI_PYTHON") or "python"
local python_cmd = python_path .. " -c ..."
```

## Recommended Installation Steps

### 1. Install Python Package

```bash
# Choose installation location (example)
cd ~/projects  # or C:\Users\YourName\

# Clone repository
git clone https://github.com/your-org/fusion_ai.git
cd fusion_ai

# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install package
pip install -e ".[all]"
```

### 2. Install Fusion Fuses

**Option A: Automatic (with venv activated)**
```bash
python scripts/install_fuses.py
```

**Option B: Manual**
Copy all `.fuse` files from `fusion_ai/fuses/` to Fusion's Fuses directory.

### 3. Configure Python Path

**Edit each .fuse file** and update the Python command:

Find this pattern in each .fuse:
```lua
local python_cmd = string.format(
    'python -c "from fusion_ai.fuses.fusion_bridge import ...',
```

Replace with absolute path:
```lua
-- Linux/macOS example
local python_path = "/home/yourname/projects/fusion_ai/venv/bin/python"

-- Windows example
local python_path = "C:/Users/YourName/fusion_ai/venv/Scripts/python.exe"

local python_cmd = string.format(
    '%s -c "from fusion_ai.fuses.fusion_bridge import ...',
    python_path,
```

## Verifying Installation

### 1. Test Python Package
```bash
# Activate venv
source venv/bin/activate

# Test import
python -c "import fusion_ai; print('Success!')"
python -c "from fusion_ai.models import DepthAnythingV2; print('Models OK!')"
```

### 2. Test Fusion Integration

1. Open Fusion or DaVinci Resolve
2. Look for "AI Tools" category in node list
3. Add a node (e.g., DepthAnythingV2)
4. Connect to a Loader
5. Render a frame

If you see output, it's working! If not, check:
- Python path in .fuse files is correct
- venv has fusion_ai installed
- Models can be imported

## Troubleshooting

### "No module named 'fusion_ai'"

**Problem:** Fusion's Python can't find the package.

**Solution:**
1. Verify venv installation: `pip show fusion-ai`
2. Update Python path in .fuse files to point to venv
3. Check that fusion_ai is installed in that venv

### "python: command not found" (Linux/macOS)

**Problem:** Python not in PATH or venv not activated.

**Solution:**
- Use absolute path in .fuse files: `/full/path/to/venv/bin/python`

### "'python' is not recognized" (Windows)

**Problem:** Python not in PATH.

**Solution:**
- Use absolute path in .fuse files: `C:/path/to/venv/Scripts/python.exe`
- Use forward slashes `/` even on Windows (Lua prefers this)

### Models downloading to wrong location

**Problem:** Models downloading to system location instead of project cache.

**Solution:**
```bash
# Set cache directory
export FUSION_AI_CACHE_DIR="/path/to/fusion_ai/.cache"
# OR set in .fuse files
```

## Example Complete Setup

```bash
# 1. Install Python package
cd ~/projects
git clone https://github.com/your-org/fusion_ai.git
cd fusion_ai
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"

# 2. Note the Python path
which python
# Output: /home/yourname/projects/fusion_ai/venv/bin/python

# 3. Install fuses
python scripts/install_fuses.py

# 4. Edit fuses to use absolute Python path
cd ~/.fusion/BlackmagicDesign/Fusion/Fuses/
# Edit each .fuse file to use the path from step 2

# 5. Launch Fusion
# Fusion will now use your venv Python to run AI models
```

## Best Practices

1. **Keep venv with project**: Don't move fusion_ai folder after setup
2. **Use absolute paths**: Hardcode full path to venv Python in fuses
3. **Document your paths**: Keep note of where you installed
4. **Test after updates**: Verify fuses still work after git pull
5. **Cache models once**: Point all fuses to same cache directory

## For Multiple Users / Shared Systems

If installing for multiple users:

```bash
# System-wide installation location
/opt/fusion_ai/
├── venv/
└── fusion_ai/

# Each user's fuses point to same venv
# Fuses location (per user):
~/.fusion/BlackmagicDesign/Fusion/Fuses/

# Set permissions
sudo chown -R fusion-admin:users /opt/fusion_ai
sudo chmod -R 755 /opt/fusion_ai
```

## Summary

**Key Points:**
- ✅ Python package can be **anywhere** (your choice)
- ✅ Fuses **must** be in Fusion's Fuses directory
- ✅ Edit fuses to use **absolute path** to venv Python
- ✅ Test that Fusion can import fusion_ai
- ✅ All models will cache to FUSION_AI_CACHE_DIR

**The locations are independent** - Fusion just needs to know where to find your venv Python!
