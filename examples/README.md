# Fusion AI Examples

This directory contains examples demonstrating how to use the Fusion AI package.

## Python Examples

### example_depth.py
Demonstrates DepthAnythingV2 usage:
- Basic depth map generation
- Colored depth maps with different colormaps
- Batch processing multiple images
- Model size comparison
- High-precision depth (EXR format)

**Run:**
```bash
python examples/example_depth.py
```

### example_qwen.py
Demonstrates Qwen VL usage:
- Image description
- Visual question answering
- Edit instructions
- Custom prompts
- Image analysis

**Run:**
```bash
python examples/example_qwen.py
```

## Fusion Workflow Guide

### fusion_workflow.md
Complete guide for using Fusion AI in Blackmagic Fusion:
- Basic workflows for depth-based compositing
- AI-assisted rotoscoping
- Content-aware analysis
- Batch processing
- Advanced techniques
- Performance optimization
- Troubleshooting

## Quick Start

### 1. Basic Depth Map

```python
from fusion_ai.models import DepthAnythingV2

model = DepthAnythingV2(model_size="base")
depth_map = model.infer("your_image.jpg", normalize=True)
model.save_depth(depth_map, "depth_output.png")
```

### 2. Image Description

```python
from fusion_ai.models import QwenEdit

model = QwenEdit()
description = model.describe("your_image.jpg")
print(description)
```

### 3. In Fusion

1. Add "Depth Anything V2" node from AI Tools category
2. Connect your footage
3. Adjust parameters
4. Use depth output for effects

## Example Images

The examples use placeholder paths. Replace with your own images:

```python
# Replace this:
model.infer("path/to/your/image.jpg")

# With actual path:
model.infer("/home/user/my_photos/image.jpg")
```

## Common Use Cases

### VFX and Compositing
- Generate depth maps for 3D effects
- Create depth-based color grading
- Assist with rotoscoping
- Generate fog and atmosphere

### Content Analysis
- Automatic scene description
- Shot logging
- Quality control
- Content tagging

### Workflow Automation
- Batch process footage
- Generate metadata
- Create proxies with depth info

## Performance Notes

### Model Download
First run will download models:
- DepthAnythingV2: ~400MB per model size
- Qwen VL: ~10GB

Models are cached, so subsequent runs are fast.

### GPU Usage
For best performance:
```python
# Use CUDA if available
model = DepthAnythingV2(device="cuda")

# Or force CPU
model = DepthAnythingV2(device="cpu")
```

### Memory Requirements

| Model | Size | GPU Memory | Quality |
|-------|------|------------|---------|
| DepthAnythingV2 Small | ~180MB | ~2GB | Good |
| DepthAnythingV2 Base | ~380MB | ~4GB | Better |
| DepthAnythingV2 Large | ~1.3GB | ~8GB | Best |
| Qwen VL | ~10GB | ~16GB | - |

## Troubleshooting

### ImportError
```bash
# Install the package
pip install -e .
```

### CUDA Out of Memory
```python
# Try smaller model
model = DepthAnythingV2(model_size="small")

# Or use CPU
model = DepthAnythingV2(device="cpu")
```

### Model Download Issues
```python
# Set custom cache directory
from fusion_ai.config import CACHE_DIR
print(f"Models cached at: {CACHE_DIR}")
```

## Next Steps

1. Try the basic examples with your own images
2. Read the [Fusion Workflow Guide](fusion_workflow.md)
3. Check the main [README](../README.md) for API reference
4. Explore the [fuses directory](../fusion_ai/fuses/) for Fusion integration

## Contributing

Have a great example? Submit a PR with:
- Well-commented code
- Clear use case description
- Sample output (if applicable)
