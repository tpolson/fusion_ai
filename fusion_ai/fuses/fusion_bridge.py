"""Bridge between Fusion and Python AI models."""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion_ai.models.depth_anything_v2 import DepthAnythingV2
from fusion_ai.models.qwen_edit import QwenEdit


class FusionBridge:
    """Bridge class for calling AI models from Fusion."""

    def __init__(self):
        """Initialize the bridge."""
        self.models = {}

    def load_depth_model(self, model_size: str = "base", device: str = "cuda") -> str:
        """Load DepthAnythingV2 model.

        Args:
            model_size: Model size (small/base/large)
            device: Device to use

        Returns:
            Status message
        """
        try:
            model_id = f"depth_{model_size}"
            if model_id not in self.models:
                self.models[model_id] = DepthAnythingV2(
                    model_size=model_size,
                    device=device
                )
            return json.dumps({"status": "success", "message": f"Loaded {model_id}"})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def load_qwen_model(self, device: str = "cuda") -> str:
        """Load Qwen Edit model.

        Args:
            device: Device to use

        Returns:
            Status message
        """
        try:
            model_id = "qwen_edit"
            if model_id not in self.models:
                self.models[model_id] = QwenEdit(device=device)
            return json.dumps({"status": "success", "message": f"Loaded {model_id}"})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def process_depth(
        self,
        input_path: str,
        output_path: str,
        model_size: str = "base",
        normalize: bool = True,
        colormap: Optional[str] = None
    ) -> str:
        """Process image for depth estimation.

        Args:
            input_path: Input image path
            output_path: Output depth map path
            model_size: Model size to use
            normalize: Whether to normalize output
            colormap: Optional colormap name

        Returns:
            Status message
        """
        try:
            # Load model if not loaded
            model_id = f"depth_{model_size}"
            if model_id not in self.models:
                self.load_depth_model(model_size)

            model = self.models[model_id]

            # Run inference
            depth = model.infer(
                input_path,
                normalize=normalize,
                colormap=colormap
            )

            # Save output
            Image.fromarray(depth).save(output_path)

            return json.dumps({
                "status": "success",
                "message": "Depth map generated",
                "output_path": output_path
            })

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def process_qwen(
        self,
        input_path: str,
        prompt: str,
        mode: str = "describe"
    ) -> str:
        """Process image with Qwen model.

        Args:
            input_path: Input image path
            prompt: Text prompt
            mode: Processing mode (describe/answer/edit)

        Returns:
            Model response
        """
        try:
            # Load model if not loaded
            model_id = "qwen_edit"
            if model_id not in self.models:
                self.load_qwen_model()

            model = self.models[model_id]

            # Run inference based on mode
            if mode == "describe":
                result = model.describe(input_path)
            elif mode == "answer":
                result = model.answer(input_path, prompt)
            elif mode == "edit":
                result = model.edit(input_path, prompt)
            else:
                result = model.infer(input_path, prompt)

            return json.dumps({
                "status": "success",
                "message": "Processing complete",
                "result": result
            })

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


# Global bridge instance
_bridge = FusionBridge()


def depth_anything_v2_process(
    input_path: str,
    output_path: str,
    model_size: str = "base",
    normalize: bool = True,
    colormap: Optional[str] = None
) -> str:
    """Process depth estimation - callable from Fusion."""
    return _bridge.process_depth(input_path, output_path, model_size, normalize, colormap)


def qwen_process(
    input_path: str,
    prompt: str,
    mode: str = "describe"
) -> str:
    """Process with Qwen - callable from Fusion."""
    return _bridge.process_qwen(input_path, prompt, mode)


if __name__ == "__main__":
    # CLI interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Fusion AI Bridge")
    parser.add_argument("--model", choices=["depth", "qwen"], required=True)
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", help="Output path (for depth)")
    parser.add_argument("--prompt", help="Text prompt (for qwen)")
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--mode", default="describe", choices=["describe", "answer", "edit"])

    args = parser.parse_args()

    if args.model == "depth":
        result = depth_anything_v2_process(
            args.input,
            args.output or "output_depth.png",
            args.model_size
        )
    else:
        result = qwen_process(
            args.input,
            args.prompt or "Describe this image",
            args.mode
        )

    print(result)
