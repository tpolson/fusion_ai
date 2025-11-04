# Contributing to Fusion AI

Thank you for your interest in contributing to Fusion AI! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs

### Suggesting Features

Feature suggestions are welcome! Please include:
- Use case description
- Proposed solution or API
- Why this would be useful
- Any implementation ideas

### Contributing Code

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/fusion_ai.git
   cd fusion_ai
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make Changes**
   - Write code following our style guide
   - Add tests for new features
   - Update documentation

5. **Run Tests**
   ```bash
   pytest tests/
   ```

6. **Submit Pull Request**
   - Clear description of changes
   - Link to related issues
   - Screenshots if UI changes

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- CUDA toolkit (optional, for GPU support)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fusion_ai.git
cd fusion_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fusion_ai --cov-report=html

# Run specific test file
pytest tests/test_depth_anything_v2.py
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black fusion_ai/

# Sort imports
isort fusion_ai/

# Lint code
flake8 fusion_ai/

# Type checking
mypy fusion_ai/
```

## Code Style Guide

### Python Style

Follow PEP 8 with these specifics:
- Line length: 100 characters
- Use type hints for function signatures
- Docstrings in Google style

**Example:**

```python
def process_image(
    image: Union[str, Path, Image.Image],
    normalize: bool = True,
    device: str = "cuda"
) -> np.ndarray:
    """Process an image for depth estimation.

    Args:
        image: Input image (path or PIL Image)
        normalize: Whether to normalize output
        device: Device to use for inference

    Returns:
        Processed depth map as numpy array

    Raises:
        ValueError: If image format is not supported
    """
    pass
```

### Lua/Fuse Style

For Fusion Fuse plugins:
- Use 4-space indentation
- Comment complex logic
- Follow existing Fuse patterns

## Project Structure

```
fusion_ai/
├── fusion_ai/          # Main package
│   ├── core/           # Core functionality
│   ├── models/         # Model implementations
│   ├── fuses/          # Fusion plugins
│   └── utils/          # Utilities
├── tests/              # Test files
├── examples/           # Usage examples
├── scripts/            # Setup and install scripts
└── docs/               # Documentation
```

## Adding New Models

To add a new AI model:

1. **Create Model Class**
   ```python
   # fusion_ai/models/your_model.py
   from fusion_ai.core.base_model import BaseModel

   class YourModel(BaseModel):
       def load_model(self):
           # Load your model
           pass

       def infer(self, input_data, **kwargs):
           # Run inference
           pass
   ```

2. **Update Configuration**
   ```python
   # fusion_ai/config.py
   MODEL_CONFIGS["your_model"] = {
       "default": {
           "repo_id": "huggingface/model-name",
           # ... other config
       }
   }
   ```

3. **Create Fuse Plugin** (optional)
   ```lua
   -- fusion_ai/fuses/YourModel.fuse
   FuRegisterClass("YourModel", CT_Tool, {
       REGS_Name = "Your Model",
       REGS_Category = "AI Tools",
       -- ... implement
   })
   ```

4. **Add Tests**
   ```python
   # tests/test_your_model.py
   def test_your_model_initialization():
       model = YourModel()
       assert model is not None
   ```

5. **Add Example**
   ```python
   # examples/example_your_model.py
   from fusion_ai.models import YourModel
   # ... usage example
   ```

6. **Update Documentation**
   - Add to README.md
   - Create usage guide
   - Document parameters

## Testing Guidelines

### Unit Tests
- Test each model individually
- Mock external dependencies
- Test error handling

### Integration Tests
- Test model loading
- Test inference with sample data
- Test Fusion integration

### Example Test

```python
import pytest
from fusion_ai.models import DepthAnythingV2

def test_depth_model_initialization():
    model = DepthAnythingV2(model_size="small")
    assert model.model_size == "small"
    assert model.device in ["cuda", "cpu"]

def test_depth_inference():
    # Use a small test image
    model = DepthAnythingV2(model_size="small", device="cpu")
    depth = model.infer("tests/fixtures/test_image.jpg")
    assert depth is not None
    assert depth.shape[0] > 0
```

## Documentation

### Docstrings
All public functions should have docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When parameter is invalid
    """
```

### README Updates
When adding features, update:
- Feature list
- Usage examples
- API reference
- Requirements

## Pull Request Process

1. **Before Submitting**
   - Run all tests
   - Update documentation
   - Add examples if applicable
   - Run code quality tools

2. **PR Description**
   - What changed and why
   - Related issues
   - Testing performed
   - Breaking changes (if any)

3. **Review Process**
   - Maintainers will review
   - Address feedback
   - Squash commits if requested

4. **After Merge**
   - Delete branch
   - Close related issues

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Give constructive feedback
- Credit others' work

## Questions?

- Open an issue for questions
- Tag with "question" label
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors will be added to:
- README.md contributors section
- Release notes
- Project credits

Thank you for contributing! 🎉
