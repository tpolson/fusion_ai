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
        colormap: Optional[str] = None,
        use_fp16: bool = False,
        output_format: str = "auto",
        alpha_path: Optional[str] = None
    ) -> str:
        """Process image for depth estimation.

        Args:
            input_path: Input image path
            output_path: Output depth map path
            model_size: Model size to use
            normalize: Whether to normalize output
            colormap: Optional colormap name
            use_fp16: Use fp16 precision (default: fp32)
            output_format: Output format (auto/exr/png) - auto uses exr for fp16/fp32
            alpha_path: Optional path to save alpha channel separately

        Returns:
            Status message with paths
        """
        try:
            # Determine output dtype based on format and precision
            if output_format == "exr" or (output_format == "auto" and not colormap):
                output_dtype = np.float16 if use_fp16 else np.float32
                use_exr = True
            else:
                output_dtype = np.uint8
                use_exr = False

            # Load model if not loaded
            model_id = f"depth_{model_size}_{'fp16' if use_fp16 else 'fp32'}"
            if model_id not in self.models:
                self.models[model_id] = DepthAnythingV2(
                    model_size=model_size,
                    use_fp16=use_fp16
                )

            model = self.models[model_id]

            # Run inference
            result = model.infer(
                input_path,
                normalize=normalize,
                colormap=colormap,
                output_dtype=output_dtype,
                preserve_alpha=True
            )

            # Handle result (might be depth only or (depth, alpha) tuple)
            if isinstance(result, tuple):
                depth, alpha = result
                has_alpha = True
            else:
                depth = result
                has_alpha = False

            # Save depth output
            if use_exr and not colormap:
                from fusion_ai.utils.image import save_exr
                save_exr(depth, output_path)
            else:
                if depth.dtype in (np.float16, np.float32):
                    # Convert float to uint8 for PNG
                    from fusion_ai.utils.image import float_to_image
                    depth = float_to_image(depth, output_dtype=np.uint8)
                Image.fromarray(depth).save(output_path)

            # Save alpha channel if present and requested
            alpha_saved = None
            if has_alpha and alpha_path:
                # Save alpha as EXR to preserve precision
                alpha_float = alpha.astype(np.float32) / 255.0 if alpha.dtype == np.uint8 else alpha
                from fusion_ai.utils.image import save_exr
                save_exr(alpha_float, alpha_path, channels=['A'])
                alpha_saved = alpha_path

            return json.dumps({
                "status": "success",
                "message": "Depth map generated",
                "output_path": output_path,
                "alpha_path": alpha_saved,
                "format": "exr" if use_exr else "png",
                "dtype": str(output_dtype)
            })

        except Exception as e:
            import traceback
            return json.dumps({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })

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
    colormap: Optional[str] = None,
    use_fp16: bool = False,
    output_format: str = "auto",
    alpha_path: Optional[str] = None
) -> str:
    """Process depth estimation - callable from Fusion."""
    return _bridge.process_depth(
        input_path, output_path, model_size, normalize, colormap,
        use_fp16, output_format, alpha_path
    )


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
