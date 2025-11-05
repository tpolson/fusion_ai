# Fusion AI - AI Models for Blackmagic Design Fusion

A comprehensive Python package integrating state-of-the-art AI models into Blackmagic Design Fusion for professional compositing, VFX, and video workflows.

## 🎯 Features

- **13 AI-Powered Fusion Nodes** - Native Fuse plugins for seamless integration
- **EXR/FP16/FP32 Pipeline** - Professional floating-point image processing
- **Temporal Consistency** - Frame-to-frame coherent video processing
- **LoRA Support** - Custom style adaptation for personalization
- **ControlNet** - Pose/edge/depth-guided generation
- **GPU Accelerated** - CUDA and Metal Performance Shaders support
- **Virtual Environment** - Isolated Python dependencies

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all features
pip install -e ".[all]"

# Install Fusion fuses
python scripts/install_fuses.py
```

**See [INSTALL.md](INSTALL.md) for complete installation instructions.**

## 🎬 Available AI Nodes

### Image Understanding
- **QwenEdit** - Vision-language model for image understanding and editing
- **DepthAnythingV2** - State-of-the-art monocular depth estimation

### Image Generation & Editing
- **AIControlNet** - ControlNet with OpenPose, Canny, Depth, MLSD, Normal, Scribble, Segmentation
- **AIStyleTransfer** - Neural style transfer with VGG19 or Stable Diffusion + LoRA

### Inpainting (Object Removal)
- **AIInpaint** - Basic inpainting (LaMa or Stable Diffusion)
- **AITemporalInpaint** - Temporally consistent object removal for video
- **AIWANInpaint** - Advanced WAN inpainting with occlusion detection

### Outpainting (Image Extension)
- **AIOutpaint** - Extend images beyond borders
- **AITemporalOutpaint** - Temporally consistent image extension for video

### Upscaling
- **AIUpscale** - AI super-resolution (Real-ESRGAN or SD x4 Upscaler)

### Video Generation
- **AIVideoGenerate** - WAN video generation with temporal consistency
- **AIFrameExtend** - Create extra frames at shot beginning/end

## 🎨 Key Capabilities

### ControlNet - Pose-Driven Animation
Track person's movement with OpenPose, generate animated character performing same motions:
```
Video → OpenPose → ControlNet → Stylized Animation
```

### WAN Inpainting - Professional Object Removal
Remove objects from video with advanced temporal consistency:
- Forward-backward optical flow
- Occlusion detection and handling
- Confidence-based warping
- Film-quality results

### LoRA Style Adaptation
Apply custom LoRA weights for personalized styles:
- Style transfer with custom artistic styles
- Character-specific adaptations
- Multiple LoRAs simultaneously

### Temporal Consistency
Frame-to-frame coherent processing for video:
- Optical flow-based warping
- Temporal smoothing
- State management for sequences

## 📋 Supported Models

### Vision-Language Models
- **Qwen-VL** - Multi-modal understanding and editing
- **Qwen2-VL** - Next-generation vision-language model

### Depth Estimation
- **DepthAnythingV2** - Sizes: Small, Base, Large
- Adjustable depth range, contrast, gamma
- Temporal smoothing for sequences

### Generative Models
- **Stable Diffusion 2.1** - Image generation
- **Stable Diffusion XL** - Higher quality generation
- **ControlNet SD 1.5** - Conditional generation (7 modalities)

### Inpainting
- **LaMa** - Fast, high-quality inpainting
- **SD 2.0 Inpainting** - Prompt-guided removal

### Upscaling
- **Real-ESRGAN** - Variants: x2, x4, x4plus, anime
- **SD x4 Upscaler** - Prompt-guided 4x upscaling

### Style Transfer
- **VGG19** - Classic neural style transfer
- **SD Img2Img** - Prompt-based style with LoRA

### Optical Flow
- **Farneback** - Dense optical flow for temporal consistency
- Forward-backward consistency checking

## 🖼️ Example Workflows

### Depth-Based Compositing
```
Loader → DepthAnythingV2 → Saver
```
- Adjust depth range, contrast, gamma in inspector
- Use depth maps for fog, DOF, relighting

### Object Removal in Video
```
Loader → AIWANInpaint ← Mask → Saver
```
- Draw mask over object to remove
- Enable occlusion detection
- Set temporal consistency to 0.85

### Character Animation from Pose
```
Video → AIControlNet (OpenPose) → Stylized Character
```
- Auto-extracts pose from input video
- Enter character description in prompt
- Outputs animated character matching movements

### AI Upscaling
```
Loader → AIUpscale → Saver
```
- Choose Real-ESRGAN (fast) or SD (quality)
- Select model variant (anime, photo, etc.)
- 2x or 4x upscaling

### Style Transfer with LoRA
```
Content Image → AIStyleTransfer ← Style Image
```
- Neural style transfer or SD-based
- Load custom LoRA weights
- Adjust style strength

## 🔧 Configuration

### Model Cache Location
```bash
export FUSION_AI_CACHE_DIR="$HOME/.cache/fusion_ai"
```

### Device Selection
```bash
export FUSION_AI_DEVICE="cuda"  # or "cpu" or "mps"
```

### Precision
All nodes support FP32 and FP16 precision selection in inspector.

## 📁 Project Structure

```
fusion_ai/
├── fusion_ai/
│   ├── models/              # AI model implementations
│   │   ├── depth_anything_v2.py
│   │   ├── qwen_edit.py
│   │   ├── inpainting.py
│   │   ├── outpainting.py
│   │   ├── upscale.py
│   │   ├── style_transfer.py
│   │   ├── frame_extension.py
│   │   ├── temporal_consistency.py
│   │   ├── video_generation.py
│   │   ├── controlnet.py
│   │   └── ...
│   ├── fuses/               # Fusion Fuse plugins (.fuse)
│   │   ├── DepthAnythingV2.fuse
│   │   ├── QwenEdit.fuse
│   │   ├── AIControlNet.fuse
│   │   ├── AIWANInpaint.fuse
│   │   ├── fusion_bridge.py  # Python bridge for Fusion
│   │   └── ...
│   ├── utils/               # Utility functions
│   │   ├── image.py         # EXR, fp16/fp32 handling
│   │   ├── lora.py          # LoRA management
│   │   └── ...
│   └── core/                # Base classes
│       └── base_model.py
├── scripts/                 # Setup and utility scripts
├── examples/                # Usage examples
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## 🎓 Usage Examples

### Python API

```python
from fusion_ai.models import DepthAnythingV2, ControlNetSD, WANInpainting

# Depth estimation
depth_model = DepthAnythingV2(model_size='base', use_fp16=True)
depth_map = depth_model.infer('image.jpg')

# ControlNet generation
controlnet = ControlNetSD(controlnet_type='openpose')
result = controlnet.infer(
    prompt="anime character, detailed",
    control_image="pose.jpg",
    seed=42
)

# WAN inpainting
wan = WANInpainting(method='sd')
inpainted = wan.infer(
    image='frame.exr',
    mask='mask.exr',
    temporal_consistency=0.85,
    detect_occlusions=True
)
```

### Fusion Workflow

1. Open Fusion or DaVinci Resolve Fusion page
2. Add AI nodes from **"AI Tools"** category
3. Connect to Loader/Saver or other nodes
4. Adjust parameters in inspector
5. Render

## ⚙️ Requirements

- **Python:** 3.10+ (3.8-3.9 no longer supported)
- **PyTorch:** >= 2.0.0 (with CUDA for GPU)
- **CUDA:** 11.8+ (for NVIDIA GPUs)
- **VRAM:** 8GB minimum, 16GB+ recommended
- **Disk Space:** ~50GB for all models cached
- **Fusion/Resolve:**
  - Fusion Studio 9+ (tested with **v20**)
  - DaVinci Resolve 16+ (tested with **v20**)

## 🐛 Troubleshooting

### CUDA out of memory
- Use FP16 instead of FP32
- Reduce image resolution
- Use smaller model variants

### Fusion can't find fuses
- Verify fuses in correct directory
- Restart Fusion/Resolve
- Check venv is activated

### Models downloading slowly
- First run downloads models (can be large)
- Models are cached for subsequent use
- Check internet connection

**See [INSTALL.md](INSTALL.md) for detailed troubleshooting.**

## 📚 Documentation

- **[INSTALL.md](INSTALL.md)** - Complete installation guide with venv setup
- **[examples/](examples/)** - Example scripts and workflows
- **[docs/](docs/)** - API documentation and guides

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Models

1. Create model class in `fusion_ai/models/`
2. Inherit from `BaseModel`
3. Implement `load_model()` and `infer()` methods
4. Create Lua fuse in `fusion_ai/fuses/`
5. Add bridge function in `fusion_bridge.py`

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **Qwen-VL** - Alibaba Cloud
- **DepthAnythingV2** - TikTok/ByteDance
- **Stable Diffusion** - Stability AI
- **ControlNet** - Lvmin Zhang
- **Real-ESRGAN** - Tencent ARC Lab
- **LaMa** - Samsung AI
- Built for **Blackmagic Design Fusion**

## 💬 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Documentation:** See `docs/` directory

## 🎯 Roadmap

- [ ] SDXL ControlNet support
- [ ] AnimateDiff integration (on request)
- [ ] Additional ControlNet modalities
- [ ] Real-time preview optimization
- [ ] Batch processing utilities
- [ ] More LoRA presets

---

**Made with ❤️ for the Fusion community**
