# Fusion AI Workflow Guide

This guide shows how to use Fusion AI tools in your Blackmagic Fusion workflows.

## Prerequisites

1. Blackmagic Fusion 9+ or DaVinci Resolve 16+
2. Python 3.8+ installed and in PATH
3. fusion-ai package installed
4. Fuse plugins installed (run `python scripts/install_fuses.py`)

## Basic Workflows

### Workflow 1: Depth-Based Compositing

Create depth maps for 3D-like effects and compositing.

**Steps:**

1. **Add Nodes:**
   - Loader (your footage)
   - DepthAnythingV2 (AI Tools > Depth Anything V2)
   - Displace or other depth-based effects

2. **Configure DepthAnythingV2:**
   - Model Size: Base (good balance of quality and speed)
   - Normalize: Checked
   - Colormap: None (for depth data) or Inferno (for visualization)
   - Device: CUDA (if available)

3. **Use Depth Map:**
   - Connect depth output to Displace's Displacement Input
   - Or use with Fog3D, DepthBlur, etc.

**Use Cases:**
- Depth-based color grading
- Fake 3D effects
- Selective focus/blur
- Fog and atmosphere effects

### Workflow 2: AI-Assisted Rotoscoping

Use depth maps to assist with rotoscoping.

**Steps:**

1. **Generate Depth:**
   - Loader → DepthAnythingV2
   - Use depth to identify foreground/background

2. **Create Mask:**
   - Use depth map in Matte Control
   - Refine with traditional roto tools

3. **Apply Effects:**
   - Merge foreground and background with different treatments

### Workflow 3: Content-Aware Analysis

Use Qwen VL to analyze and understand your footage.

**Steps:**

1. **Add QwenEdit Node:**
   - AI Tools > Qwen Edit

2. **Configure:**
   - Mode: Describe or Answer Question
   - Prompt: "Describe the scene composition and lighting"
   - Device: CUDA

3. **Review Results:**
   - Check console output or Text output
   - Use insights for color grading, effects decisions

**Use Cases:**
- Shot analysis and logging
- Quality control
- Automatic scene detection
- Content tagging

### Workflow 4: Batch Processing

Process entire sequences efficiently.

**Python Script for Fusion:**

```python
# In Fusion's Python console or script
from fusion_ai.models import DepthAnythingV2
import os

# Get composition
comp = fusion.CurrentComp

# Initialize model
depth_model = DepthAnythingV2(model_size='base')

# Process frame range
for frame in range(1001, 1100):
    # Set current time
    comp.CurrentTime = frame

    # Get frame from loader
    loader = comp.Loader1
    frame_path = loader.GetAttrs()["TOOLST_Clip_Name"][1]

    # Generate depth
    depth = depth_model.infer(frame_path)

    # Save depth map
    output_path = f"depth_maps/frame_{frame:04d}.png"
    depth_model.save_depth(depth, output_path)

    print(f"Processed frame {frame}")
```

## Advanced Techniques

### Technique 1: Depth-Based Lighting

Use depth information to add realistic lighting effects.

1. Generate depth map
2. Use depth to drive light intensity (near = brighter)
3. Apply with Color Curves or Custom tool

### Technique 2: Automatic Foreground Extraction

Combine depth and edge detection for automatic mattes.

1. DepthAnythingV2 → depth map
2. Edge Detect on original image
3. Combine both for refined matte
4. Clean up with Matte Control

### Technique 3: AI-Guided Color Grading

Use Qwen VL to analyze scenes and suggest grading.

1. Use QwenEdit to analyze mood and lighting
2. Prompt: "Describe the color palette and suggest improvements"
3. Apply suggested adjustments manually or with LUTs

## Performance Tips

### GPU Usage

- Always use CUDA device for real-time work
- Monitor GPU memory with nvidia-smi
- Use smaller models if memory limited

### Caching

```python
# Cache models in Fusion script
from fusion_ai.models import DepthAnythingV2

# Load once at start
if not hasattr(comp, '_depth_model'):
    comp._depth_model = DepthAnythingV2(model_size='base')

# Reuse throughout composition
depth = comp._depth_model.infer(image_path)
```

### Batch vs. Real-Time

**Real-Time (Interactive):**
- Use "small" models
- Lower resolution
- Cache aggressively

**Batch (Final Render):**
- Use "large" models
- Full resolution
- Process overnight

## Integration Patterns

### Pattern 1: Saver Integration

Save depth maps alongside renders:

```lua
-- In Fusion comp
Loader1 -> DepthAnythingV2 -> Saver1
                          |-> Merge with original
```

### Pattern 2: Conditional Processing

Only process frames that need it:

```python
import os

def process_if_needed(frame_path, depth_path, model):
    if not os.path.exists(depth_path):
        depth = model.infer(frame_path)
        model.save_depth(depth, depth_path)
    return depth_path
```

### Pattern 3: Multi-Pass Workflow

1. **Pass 1:** Generate all depth maps
2. **Pass 2:** Use depth maps for effects
3. **Pass 3:** Final composite and grade

## Troubleshooting

### "Python not found"
- Add Python to system PATH
- Restart Fusion after PATH changes

### "Out of memory"
- Use smaller model size
- Reduce input resolution
- Close other GPU applications

### Slow Performance
- Ensure CUDA device is selected
- Check GPU utilization (should be 80%+)
- Consider batch processing instead of real-time

### Fuses Not Appearing
- Verify installation: `python scripts/install_fuses.py --check`
- Restart Fusion completely
- Check Fusion logs for errors

## Best Practices

1. **Test First:** Try on a single frame before batch processing
2. **Save Intermediate Results:** Don't regenerate depth maps every time
3. **Use Appropriate Model Sizes:** Balance quality vs. speed
4. **Cache Models:** Load once, use many times
5. **Monitor Resources:** Watch GPU memory and temperature

## Example Compositions

### Composition 1: Depth-Enhanced Grading

```
Loader
  ├─> DepthAnythingV2
  │     └─> ColorCurves (depth-based)
  └─> Merge (combine with original)
        └─> ColorCorrector (final grade)
              └─> Saver
```

### Composition 2: AI Analysis Pipeline

```
Loader
  ├─> QwenEdit (describe)
  │     └─> Text Output
  └─> DepthAnythingV2
        └─> Saver (depth maps)
```

## Further Resources

- [Python API Documentation](../README.md)
- [Model Specifications](../fusion_ai/config.py)
- [Fuse Development](../fusion_ai/fuses/README.md)
