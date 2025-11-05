# New AI Nodes Architecture

## Overview

Fusion AI now has a modular architecture with separate nodes for each AI function. All nodes use EXR format for fp16/fp32 precision and support alpha channels.

## Implemented Nodes

### 1. DepthAnythingV2 ✅ (Complete with Fuse)
**Purpose**: Generate depth maps from images/sequences

**Inputs**:
- Image (EXR/PNG)

**Outputs**:
- Depth Map (EXR/PNG)
- Alpha Channel (optional)

**Inspector Controls**:
- Model Size (Small/Base/Large)
- Precision (FP16/FP32)
- Output Format (Auto/EXR/PNG)
- Normalize
- Depth Adjustments:
  - Invert Depth
  - Depth Min/Max
  - Contrast
  - Gamma
  - Temporal Smoothing
- Colormap (None/Inferno/Viridis/Plasma/Turbo)
- Preserve Alpha

**Workflow**: `Loader → DepthAnythingV2 → Saver`

---

### 2. QwenEdit ✅ (Complete with Fuse)
**Purpose**: AI image understanding and inpainting

**Inputs**:
- Image (EXR)
- Matte (EXR) - for inpainting

**Outputs**:
- Output Image (EXR)
- Text Result

**Modes**:
- Describe: Describe image content
- Answer Question: Answer questions about image
- Edit Instruction: Text-guided editing
- Remove Object (Inpaint): Remove objects using matte

**Inspector Controls**:
- Mode selector
- Prompt text
- Inpaint Method (LaMa/Stable Diffusion)
- Device (CUDA/CPU)

**Workflow**: `Loader + Matte → QwenEdit → Output`

---

## New Models (Python Implementation Ready)

### 3. Outpainting 🆕 (Model Ready, Fuse Needed)
**Purpose**: Extend images beyond their borders

**Model**: `StableDiffusionOutpainting`
**File**: `fusion_ai/models/outpainting.py`

**Features**:
- Extend in all directions (all/horizontal/vertical/top/bottom/left/right)
- Customizable extension pixels
- Seamless integration with original image
- Prompt-guided generation

**Planned Inputs**:
- Image (EXR)

**Planned Outputs**:
- Extended Image (EXR)

**Planned Inspector Controls**:
- Extend Pixels (slider: 64-512)
- Direction (All/Horizontal/Vertical/Top/Bottom/Left/Right)
- Prompt (optional text guidance)
- Inference Steps
- Guidance Scale

**Use Cases**:
- Expand composition
- Add missing edges
- Create wider aspect ratios
- Extend backgrounds

---

### 4. Upscale 🆕 (Models Ready, Fuse Needed)
**Purpose**: AI-based image upscaling

**Models**:
- `RealESRGAN` - High-quality 2x/4x upscaling
- `StableDiffusionUpscale` - Creative 4x upscaling

**File**: `fusion_ai/models/upscale.py`

**Features**:
- Multiple model variants:
  - RealESRGAN_x4plus (photorealistic)
  - RealESRNet_x4plus (sharper)
  - RealESRGAN_x4plus_anime_6B (anime/illustration)
  - RealESRGAN_x2plus (2x scale)
- Tile-based processing for large images
- SD Upscaler with prompt guidance

**Planned Inputs**:
- Image (EXR/PNG)

**Planned Outputs**:
- Upscaled Image (EXR/PNG)

**Planned Inspector Controls**:
- Method (Real-ESRGAN/SD Upscaler)
- Model Variant (x2plus/x4plus/x4plus_anime)
- Scale (2x/4x/custom)
- Prompt (for SD Upscaler)

**Use Cases**:
- Increase resolution for deliverables
- Upscale low-res plates
- Enhance detail
- Prepare for large format output

---

### 5. Style Transfer 🆕 (Models Ready, Fuse Needed)
**Purpose**: Apply artistic styles to images

**Models**:
- `StyleTransfer` - Neural style transfer (VGG19-based)
- `InstantStyleTransfer` - Fast arbitrary style transfer

**File**: `fusion_ai/models/style_transfer.py`

**Features**:
- Apply any reference style to content image
- Adjustable style/content weight
- Gram matrix style representation
- Optimization-based approach

**Planned Inputs**:
- Content Image (EXR)
- Style Image (EXR)

**Planned Outputs**:
- Stylized Image (EXR)

**Planned Inspector Controls**:
- Style Weight (slider: 1e5-1e7)
- Content Weight (slider: 0.1-10)
- Optimization Steps (50-500)
- Method (Neural/Instant)

**Use Cases**:
- Artistic effects
- Match look of reference images
- Creative transitions
- Style consistency across shots

---

### 6. Frame Extension 🆕 (Model Ready, Fuse Needed)
**Purpose**: Create extra frames at shot beginning/end

**Models**:
- `FrameExtension` - Temporal inpainting and motion prediction
- `FrameInterpolation` - Smooth frame interpolation

**File**: `fusion_ai/models/frame_extension.py`

**Features**:
- Optical flow-based motion estimation
- Frame warping for natural motion
- Multiple blending modes
- AI inpainting for clean results
- Works for both directions (forward/backward)

**Planned Inputs**:
- Frames (2-5 reference frames)

**Planned Outputs**:
- Extended Frames (sequence)

**Planned Inspector Controls**:
- Direction (Forward/Backward)
- Number of New Frames (1-30)
- Blend Mode (Optical Flow/Simple/Inpainting)
- Use AI Refinement (checkbox)

**Use Cases**:
- Extend shot duration
- Create handles for transitions
- Match action timing
- Fill missing footage
- Create looping sequences

**Workflow**:
```
TimeStretcher (grab last 3 frames) → FrameExtension → Append to sequence
```

---

## Node Naming Convention

All AI nodes follow this pattern:
- **Purpose-based names**: Clear what they do
- **Separate nodes**: One function per node
- **Consistent interface**: All use EXR, fp16/fp32, alpha support

**Existing**:
- DepthAnythingV2
- QwenEdit

**New (to be added)**:
- AIOutpaint
- AIUpscale
- AIStyleTransfer
- AIFrameExtend

---

## Implementation Status

| Node | Python Model | Lua Fuse | Bridge Function | Status |
|------|-------------|----------|-----------------|---------|
| DepthAnythingV2 | ✅ | ✅ | ✅ | Complete |
| QwenEdit | ✅ | ✅ | ✅ | Complete |
| AIOutpaint | ✅ | ⏳ | ⏳ | Model Ready |
| AIUpscale | ✅ | ⏳ | ⏳ | Model Ready |
| AIStyleTransfer | ✅ | ⏳ | ⏳ | Model Ready |
| AIFrameExtend | ✅ | ⏳ | ⏳ | Model Ready |

---

## Dependencies

### Core (Already Added):
- torch>=2.0.0
- transformers>=4.35.0
- diffusers>=0.21.0
- OpenEXR>=1.3.0

### Optional - Inpainting:
- simple-lama-inpainting>=0.1.0
- accelerate>=0.25.0

### Optional - Upscaling (New):
- realesrgan>=0.3.0
- basicsr>=1.4.2

### Optional - Style Transfer (New):
- torchvision>=0.15.0 (already included)

---

## Next Steps

1. ✅ Create Python models for new functions
2. ⏳ Create Lua fuses for each new node
3. ⏳ Add bridge functions in fusion_bridge.py
4. ⏳ Update dependencies in pyproject.toml
5. ⏳ Test each node in Fusion
6. ⏳ Document usage examples

---

## Benefits of Separate Nodes

1. **Clear Purpose**: Each node name tells you exactly what it does
2. **Clean Inspector**: Only relevant controls shown
3. **Modular**: Use any combination of nodes
4. **Fusion Convention**: Matches Fusion's design philosophy
5. **Easy to Learn**: One concept per node
6. **Flexible Workflows**: Mix and match as needed

---

## Example Workflows

### Depth Map Sequence:
```
Loader → DepthAnythingV2 → Saver
```

### Object Removal:
```
Loader → QwenEdit ← Matte
       → Output
```

### Upscale and Extend:
```
Loader → AIUpscale → AIOutpaint → Saver
```

### Style Transfer Sequence:
```
Loader (content) → AIStyleTransfer ← Loader (style)
                 → Saver
```

### Shot Extension:
```
Loader (last 3 frames) → AIFrameExtend → Append to timeline
```

### Combined Pipeline:
```
Loader → DepthAnythingV2 → Depth Output
      ↓
      → QwenEdit (remove object) → AIUpscale → Final Output
```
