# Fusion Fuse Plugins

This directory contains Fuse plugins for integrating AI models into Blackmagic Design Fusion.

## Available Fuses

### 1. DepthAnythingV2.fuse
Generates depth maps from input images.

**Parameters:**
- **Model Size**: Choose between Small, Base, or Large models
- **Normalize**: Normalize depth values to 0-255 range
- **Colormap**: Apply color mapping (None, Inferno, Viridis, Plasma, Turbo)
- **Device**: Select CUDA or CPU for inference

**Usage:**
1. Add the node from "AI Tools" > "Depth Anything V2"
2. Connect an image input
3. Adjust parameters as needed
4. Output is a depth map image

### 2. QwenEdit.fuse
Text-guided image understanding and editing.

**Parameters:**
- **Mode**: Select processing mode (Describe, Answer Question, Edit Instruction)
- **Prompt**: Enter text prompt or question
- **Device**: Select CUDA or CPU for inference

**Usage:**
1. Add the node from "AI Tools" > "Qwen Edit"
2. Connect an image input
3. Enter your prompt
4. Output includes pass-through image and text result

## Installation

### Automatic Installation
```bash
python scripts/install_fuses.py
```

### Manual Installation

Copy the `.fuse` files to your Fusion Fuses directory:

**Windows:**
```
%APPDATA%\Blackmagic Design\Fusion\Fuses\
```

**macOS:**
```
~/Library/Application Support/Blackmagic Design/Fusion/Fuses/
```

**Linux:**
```
~/.fusion/BlackmagicDesign/Fusion/Fuses/
```

## Python Integration

The Fuse plugins communicate with Python through the `fusion_bridge.py` module. Ensure that:

1. Python 3.8+ is installed
2. The fusion-ai package is installed (`pip install -e .`)
3. All dependencies are available
4. Python is in your system PATH

## Troubleshooting

### "Python not found" error
Ensure Python is installed and in your system PATH.

### "Module not found" error
Install the fusion-ai package:
```bash
cd /path/to/fusion_ai
pip install -e .
```

### "CUDA out of memory" error
- Try using a smaller model size
- Switch to CPU device
- Reduce input image resolution

### Models downloading slowly
Models are downloaded from Hugging Face Hub on first use. This is normal and only happens once.

## Performance Tips

1. **First Run**: Models will download on first use (may take several minutes)
2. **GPU**: Use CUDA device for much faster inference
3. **Model Size**: Start with "Base" model for good balance
4. **Caching**: Models are cached after first download

## Advanced Usage

### Using from Python in Fusion

If your Fusion version supports Python scripting directly:

```python
from fusion_ai.models import DepthAnythingV2

# In Fusion Python console
depth_model = DepthAnythingV2(model_size='base')
depth_map = depth_model.infer('path/to/image.jpg')
```

### Batch Processing

For processing multiple frames, consider using the Python API directly for better performance.
