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
from fusion_ai.models.inpainting import LamaInpainting, StableDiffusionInpainting
from fusion_ai.models.outpainting import StableDiffusionOutpainting
from fusion_ai.models.upscale import RealESRGAN, StableDiffusionUpscale
from fusion_ai.models.style_transfer import StyleTransfer, InstantStyleTransfer, StableDiffusionStyleTransfer
from fusion_ai.models.frame_extension import FrameExtension, FrameInterpolation
from fusion_ai.models.temporal_consistency import TemporalInpainting, TemporalOutpainting, WANInpainting
from fusion_ai.models.video_generation import WANVideoGenerator
from fusion_ai.models.controlnet import ControlNetSD, OpenPoseDetector, CannyDetector


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

    def load_inpainting_model(self, method: str = "lama", device: str = "cuda") -> str:
        """Load inpainting model.

        Args:
            method: Inpainting method (lama/sd)
            device: Device to use

        Returns:
            Status message
        """
        try:
            model_id = f"inpaint_{method}"
            if model_id not in self.models:
                if method == "lama":
                    self.models[model_id] = LamaInpainting(device=device)
                elif method == "sd":
                    self.models[model_id] = StableDiffusionInpainting(device=device)
                else:
                    return json.dumps({"status": "error", "message": f"Unknown method: {method}"})
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
        alpha_path: Optional[str] = None,
        invert_depth: bool = False,
        depth_min: float = 0.0,
        depth_max: float = 1.0,
        depth_contrast: float = 1.0,
        depth_gamma: float = 1.0,
        temporal_smooth: float = 0.0
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
            invert_depth: Invert depth values (near becomes far)
            depth_min: Minimum depth value (0-1)
            depth_max: Maximum depth value (0-1)
            depth_contrast: Contrast adjustment (0-3)
            depth_gamma: Gamma correction (0.1-3)
            temporal_smooth: Temporal smoothing for sequences (0-1)

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

            # Apply depth adjustments if not using colormap yet
            if not colormap:
                from fusion_ai.utils.image import adjust_depth
                depth = adjust_depth(
                    depth,
                    invert=invert_depth,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    contrast=depth_contrast,
                    gamma=depth_gamma
                )

            # TODO: Implement temporal smoothing for sequences
            # This would require caching previous frames
            if temporal_smooth > 0.0:
                # For now, just note it in the return message
                pass

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

    def process_inpaint(
        self,
        input_path: str,
        mask_path: str,
        output_path: str,
        method: str = "lama",
        prompt: Optional[str] = None,
        device: str = "cuda"
    ) -> str:
        """Process image inpainting for object removal.

        Args:
            input_path: Input image path
            mask_path: Mask image path (white = remove, black = keep)
            output_path: Output inpainted image path
            method: Inpainting method (lama/sd)
            prompt: Optional prompt for SD inpainting
            device: Device to use

        Returns:
            Status message
        """
        try:
            # Load model if not loaded
            model_id = f"inpaint_{method}"
            if model_id not in self.models:
                self.load_inpainting_model(method, device)

            model = self.models[model_id]

            # Run inpainting
            if method == "sd" and prompt:
                result = model.infer(input_path, mask_path, prompt=prompt)
            else:
                result = model.infer(input_path, mask_path)

            # Save output based on format
            if output_path.endswith('.exr'):
                from fusion_ai.utils.image import save_exr, image_to_float
                # Convert to float and save as EXR
                result_float = image_to_float(result, dtype=np.float32)
                save_exr(result_float, output_path)
            else:
                # Save as standard image format
                Image.fromarray(result).save(output_path)

            return json.dumps({
                "status": "success",
                "message": "Inpainting completed",
                "output_path": output_path,
                "method": method
            })

        except Exception as e:
            import traceback
            return json.dumps({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })


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
    alpha_path: Optional[str] = None,
    invert_depth: bool = False,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
    depth_contrast: float = 1.0,
    depth_gamma: float = 1.0,
    temporal_smooth: float = 0.0
) -> str:
    """Process depth estimation - callable from Fusion."""
    return _bridge.process_depth(
        input_path, output_path, model_size, normalize, colormap,
        use_fp16, output_format, alpha_path, invert_depth,
        depth_min, depth_max, depth_contrast, depth_gamma, temporal_smooth
    )


def qwen_process(
    input_path: str,
    prompt: str,
    mode: str = "describe"
) -> str:
    """Process with Qwen - callable from Fusion."""
    return _bridge.process_qwen(input_path, prompt, mode)


def inpaint_process(
    input_path: str,
    mask_path: str,
    output_path: str,
    method: str = "lama",
    prompt: str = ""
) -> str:
    """Process inpainting - callable from Fusion."""
    return _bridge.process_inpaint(input_path, mask_path, output_path, method, prompt if prompt else None)


def outpaint_process(
    input_path: str,
    output_path: str,
    extend_pixels: int = 256,
    direction: str = "all",
    prompt: str = "",
    inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> str:
    """Process outpainting - callable from Fusion."""
    try:
        # Load model if not loaded
        model_id = "outpaint_sd"
        if model_id not in _bridge.models:
            _bridge.models[model_id] = StableDiffusionOutpainting()

        model = _bridge.models[model_id]

        # Run outpainting
        result = model.infer(
            input_path,
            extend_pixels=extend_pixels,
            direction=direction,
            prompt=prompt if prompt else None,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale
        )

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Outpainting completed",
            "output_path": output_path
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def upscale_process(
    input_path: str,
    output_path: str,
    method: str = "realesrgan",
    model_variant: str = "RealESRGAN_x4plus",
    scale: float = 4.0,
    prompt: str = "",
    inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> str:
    """Process upscaling - callable from Fusion."""
    try:
        if method == "realesrgan":
            # Load Real-ESRGAN model
            model_id = f"upscale_{model_variant}"
            if model_id not in _bridge.models:
                _bridge.models[model_id] = RealESRGAN(model_name=model_variant)

            model = _bridge.models[model_id]

            # Run upscaling
            result = model.infer(input_path, outscale=scale)

        elif method == "sd":
            # Load SD upscaler
            model_id = "upscale_sd"
            if model_id not in _bridge.models:
                _bridge.models[model_id] = StableDiffusionUpscale()

            model = _bridge.models[model_id]

            # Run SD upscaling
            result = model.infer(
                input_path,
                prompt=prompt if prompt else "high quality, detailed",
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale
            )
        else:
            raise ValueError(f"Unknown upscale method: {method}")

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Upscaling completed",
            "output_path": output_path,
            "method": method,
            "scale": scale
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def style_transfer_process(
    content_path: str,
    style_path: str,
    output_path: str,
    method: str = "neural",
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    num_steps: int = 300,
    alpha: float = 1.0
) -> str:
    """Process style transfer - callable from Fusion."""
    try:
        if method == "neural":
            # Load neural style transfer
            model_id = "style_transfer_neural"
            if model_id not in _bridge.models:
                _bridge.models[model_id] = StyleTransfer()

            model = _bridge.models[model_id]

            # Run style transfer
            result = model.infer(
                content_path,
                style_path,
                style_weight=style_weight,
                content_weight=content_weight,
                num_steps=num_steps
            )

        elif method == "instant":
            # Load instant style transfer
            model_id = "style_transfer_instant"
            if model_id not in _bridge.models:
                _bridge.models[model_id] = InstantStyleTransfer()

            model = _bridge.models[model_id]

            # Run instant style transfer
            result = model.infer(content_path, style_path, alpha=alpha)

        else:
            raise ValueError(f"Unknown style transfer method: {method}")

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Style transfer completed",
            "output_path": output_path,
            "method": method
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def frame_extend_process(
    frame_paths: str,  # Comma-separated paths
    output_path: str,
    direction: str = "forward",
    num_frames: int = 5,
    blend_mode: str = "optical_flow",
    use_inpainting: bool = True,
    frame_index: int = 0
) -> str:
    """Process frame extension - callable from Fusion."""
    try:
        # Parse frame paths
        paths = [p.strip() for p in frame_paths.split(',')]

        # Load frame extension model
        model_id = "frame_extend"
        if model_id not in _bridge.models:
            _bridge.models[model_id] = FrameExtension()

        model = _bridge.models[model_id]

        # Run frame extension
        extended_frames = model.infer(
            paths,
            direction=direction,
            num_new_frames=num_frames,
            blend_mode=blend_mode,
            use_inpainting=use_inpainting
        )

        # Get requested frame index
        if frame_index >= len(extended_frames):
            frame_index = len(extended_frames) - 1

        result = extended_frames[frame_index]

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Frame extension completed",
            "output_path": output_path,
            "num_frames_generated": len(extended_frames),
            "frame_index": frame_index
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def temporal_inpaint_process(
    input_path: str,
    mask_path: str,
    output_path: str,
    method: str = "lama",
    temporal_consistency: float = 0.7,
    prompt: str = "",
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
    reset_state: bool = False
) -> str:
    """Process temporal inpainting - callable from Fusion."""
    try:
        # Load temporal inpainting model
        model_id = f"temporal_inpaint_{method}"
        if model_id not in _bridge.models or reset_state:
            _bridge.models[model_id] = TemporalInpainting(method=method)
            if reset_state and model_id in _bridge.models:
                _bridge.models[model_id].reset()

        model = _bridge.models[model_id]

        # Reset state if requested
        if reset_state:
            model.reset()

        # Run temporal inpainting
        result = model.infer(
            input_path,
            mask_path,
            prompt=prompt if prompt else "",
            temporal_consistency=temporal_consistency,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale
        )

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Temporal inpainting completed",
            "output_path": output_path,
            "method": method
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def temporal_outpaint_process(
    input_path: str,
    output_path: str,
    extend_pixels: int = 256,
    direction: str = "all",
    temporal_consistency: float = 0.7,
    prompt: str = "",
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
    reset_state: bool = False
) -> str:
    """Process temporal outpainting - callable from Fusion."""
    try:
        # Load temporal outpainting model
        model_id = "temporal_outpaint"
        if model_id not in _bridge.models or reset_state:
            _bridge.models[model_id] = TemporalOutpainting()
            if reset_state and model_id in _bridge.models:
                _bridge.models[model_id].reset()

        model = _bridge.models[model_id]

        # Reset state if requested
        if reset_state:
            model.reset()

        # Run temporal outpainting
        result = model.infer(
            input_path,
            extend_pixels=extend_pixels,
            direction=direction,
            prompt=prompt if prompt else "",
            temporal_consistency=temporal_consistency,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale
        )

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "Temporal outpainting completed",
            "output_path": output_path
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def wan_video_process(
    prompt: str,
    reference_frame_path: str,
    output_path: str,
    num_frames: int = 16,
    width: int = 512,
    height: int = 512,
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
    temporal_consistency: float = 0.7,
    seed: int = 42,
    frame_index: int = 0
) -> str:
    """Process WAN video generation - callable from Fusion."""
    try:
        # Load WAN video generator
        model_id = "wan_video"
        if model_id not in _bridge.models:
            _bridge.models[model_id] = WANVideoGenerator()

        model = _bridge.models[model_id]

        # Prepare reference frames if provided
        reference_frames = None
        if reference_frame_path and reference_frame_path != "":
            reference_frames = [reference_frame_path]

        # Generate video frames
        frames = model.infer(
            prompt=prompt,
            reference_frames=reference_frames,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            temporal_consistency_strength=temporal_consistency,
            seed=seed
        )

        # Get requested frame index
        if frame_index >= len(frames):
            frame_index = len(frames) - 1

        result = frames[frame_index]

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "WAN video generation completed",
            "output_path": output_path,
            "num_frames_generated": len(frames),
            "frame_index": frame_index
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def controlnet_process(
    prompt: str,
    control_image_path: str,
    output_path: str,
    controlnet_type: str = "openpose",
    negative_prompt: str = "low quality, blurry",
    conditioning_scale: float = 1.0,
    inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    auto_preprocess: bool = True,
    device: str = "cuda"
) -> str:
    """Process ControlNet generation - callable from Fusion."""
    try:
        # Load ControlNet model
        model_id = f"controlnet_{controlnet_type}"
        if model_id not in _bridge.models:
            _bridge.models[model_id] = ControlNetSD(
                controlnet_type=controlnet_type,
                device=device
            )

        model = _bridge.models[model_id]

        # Auto-preprocess if requested
        control_image = control_image_path
        if auto_preprocess and model.preprocessor:
            # Extract control signal from input image
            from fusion_ai.utils.image import save_exr, image_to_float
            preprocessed = model.preprocessor.preprocess(control_image_path)

            # Save preprocessed control image
            control_temp_path = control_image_path.replace(".exr", "_preprocessed.exr")
            result_float = image_to_float(preprocessed, dtype=np.float32)
            save_exr(result_float, control_temp_path)
            control_image = control_temp_path

        # Generate image
        result = model.infer(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
            seed=seed
        )

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        # Clean up temp file if created
        if auto_preprocess and 'control_temp_path' in locals():
            import os
            try:
                os.remove(control_temp_path)
            except:
                pass

        return json.dumps({
            "status": "success",
            "message": "ControlNet generation completed",
            "output_path": output_path,
            "controlnet_type": controlnet_type
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


def wan_inpaint_process(
    input_path: str,
    mask_path: str,
    output_path: str,
    method: str = "sd",
    temporal_consistency: float = 0.85,
    prompt: str = "",
    negative_prompt: str = "blurry, bad quality, distorted",
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
    detect_occlusions: bool = True,
    reset_state: bool = False,
    device: str = "cuda"
) -> str:
    """Process WAN inpainting - callable from Fusion."""
    try:
        # Load WAN inpainting model
        model_id = f"wan_inpaint_{method}"
        if model_id not in _bridge.models or reset_state:
            _bridge.models[model_id] = WANInpainting(method=method, device=device)
            if reset_state and model_id in _bridge.models:
                _bridge.models[model_id].reset()

        model = _bridge.models[model_id]

        # Reset state if requested
        if reset_state:
            model.reset()

        # Run WAN inpainting
        result = model.infer(
            input_path,
            mask_path,
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt,
            temporal_consistency=temporal_consistency,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            detect_occlusions=detect_occlusions
        )

        # Save output as EXR
        from fusion_ai.utils.image import save_exr, image_to_float
        result_float = image_to_float(result, dtype=np.float32)
        save_exr(result_float, output_path)

        return json.dumps({
            "status": "success",
            "message": "WAN inpainting completed",
            "output_path": output_path,
            "method": method,
            "occlusion_detection": detect_occlusions
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })


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
