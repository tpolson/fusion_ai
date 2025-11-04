# Fusion AI - AI Models for Blackmagic Design Fusion

A self-contained Python package that integrates advanced AI models into Blackmagic Design Fusion for compositing and VFX workflows.

## Features

- **Qwen Edit**: Advanced image editing and manipulation using Qwen VL models
- **DepthAnythingV2**: State-of-the-art monocular depth estimation
- **Extensible Architecture**: Easy to add more models
- **Self-Contained**: Minimal dependencies, easy installation
- **Fusion Integration**: Native Fuse plugins for seamless workflow

## Supported Models

### 1. Qwen Edit
- Image editing based on text prompts
- Visual understanding and manipulation
- Multi-modal capabilities

### 2. DepthAnythingV2
- Monocular depth estimation
- High-quality depth maps
- Multiple model sizes (Small, Base, Large)

### 3. Extensible for More Models
The architecture supports adding additional models easily.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Blackmagic Design Fusion 9+ or DaVinci Resolve 16+

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd fusion_ai

# Install the package
pip install -e .

# Install Fusion plugins
python scripts/install_fuses.py
```

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Copy Fuse files to Fusion:
   - Windows: `%APPDATA%\Blackmagic Design\Fusion\Fuses\`
   - macOS: `~/Library/Application Support/Blackmagic Design/Fusion/Fuses/`
   - Linux: `~/.fusion/BlackmagicDesign/Fusion/Fuses/`

## Usage

### In Fusion

1. Open Blackmagic Fusion or DaVinci Resolve Fusion page
2. Add nodes from the "AI Tools" category:
   - **QwenEdit**: Text-guided image editing
   - **DepthAnythingV2**: Depth map generation

### Python API

```python
from fusion_ai.models import QwenEdit, DepthAnythingV2

# Depth estimation
depth_model = DepthAnythingV2(model_size='base')
depth_map = depth_model.infer('path/to/image.jpg')

# Qwen edit
qwen = QwenEdit()
edited_image = qwen.edit('path/to/image.jpg', 'make the sky more dramatic')
```

## Project Structure

```
fusion_ai/
├── fusion_ai/           # Main Python package
│   ├── models/          # Model implementations
│   ├── fuses/           # Fusion Fuse plugins
│   ├── utils/           # Utility functions
│   └── core/            # Core functionality
├── scripts/             # Installation and setup scripts
├── examples/            # Usage examples
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- Pillow >= 9.0.0
- NumPy >= 1.21.0
- OpenCV-Python >= 4.5.0

## Configuration

Model configurations and paths can be customized in `fusion_ai/config.py` or through environment variables:

```bash
export FUSION_AI_CACHE_DIR="/path/to/model/cache"
export FUSION_AI_DEVICE="cuda"  # or "cpu"
```

## Development

### Adding New Models

1. Create a new model class in `fusion_ai/models/`
2. Inherit from `BaseModel`
3. Implement `load_model()` and `infer()` methods
4. Create corresponding Fuse plugin in `fusion_ai/fuses/`

### Testing

```bash
pytest tests/
```

## Performance Tips

- Use GPU acceleration when available
- Cache models to avoid reloading
- Batch process multiple frames when possible
- Use appropriate model sizes for your hardware

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Credits

- Qwen models by Alibaba Cloud
- DepthAnythingV2 by TikTok/ByteDance
- Built for Blackmagic Design Fusion

## Support

For issues and questions:
- GitHub Issues: <repository-issues-url>
- Documentation: See `docs/` directory
