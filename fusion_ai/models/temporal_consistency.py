"""Temporal consistency models for video inpainting and outpainting."""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class TemporalInpainting(BaseModel):
    """Temporal inpainting with frame-to-frame consistency."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize temporal inpainting model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "temporal_inpainting"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.prev_frame = None
        self.prev_mask = None
        self.load_model()

    def load_model(self) -> None:
        """Load temporal inpainting model."""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print(f"Loading temporal inpainting model...")

            model_id = "stabilityai/stable-diffusion-2-inpainting"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"Temporal inpainting loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for temporal inpainting. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load temporal inpainting model: {e}")

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute optical flow between frames.

        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)

        Returns:
            Optical flow field or None if OpenCV unavailable
        """
        try:
            import cv2

            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            return flow

        except ImportError:
            return None

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """Warp frame using optical flow.

        Args:
            frame: Frame to warp
            flow: Optical flow field

        Returns:
            Warped frame
        """
        import cv2

        h, w = frame.shape[:2]

        # Create coordinate maps
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x += flow[:, :, 0]
        y += flow[:, :, 1]

        # Remap
        warped = cv2.remap(frame, x, y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

        return warped

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "",
        temporal_consistency: float = 0.5,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> np.ndarray:
        """Perform temporal inpainting with consistency.

        Args:
            image: Input image/frame
            mask: Binary mask (white = remove, black = keep)
            prompt: Text prompt for inpainting
            temporal_consistency: Strength of temporal consistency (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            **kwargs: Additional inference parameters

        Returns:
            Inpainted image as numpy array
        """
        # Preprocess image and mask
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        pil_mask = self.preprocess_image(mask, preserve_alpha=False)

        # Convert mask to binary
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')

        # Convert to numpy
        current_frame = np.array(pil_image)
        current_mask = np.array(pil_mask)

        # Initialize with standard inpainting if no previous frame
        if self.prev_frame is None:
            # Default prompt for seamless inpainting
            if not prompt:
                prompt = "natural background, seamless fill, realistic lighting"

            # Perform inpainting
            result = self.model(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]

            result_np = np.array(result)

            # Store for next frame
            self.prev_frame = result_np.copy()
            self.prev_mask = current_mask.copy()

            return result_np

        # Temporal inpainting with previous frame
        else:
            # Compute optical flow from previous to current
            flow = self.compute_optical_flow(self.prev_frame, current_frame)

            if flow is not None and temporal_consistency > 0:
                # Warp previous result to current frame
                warped_prev = self.warp_frame(self.prev_frame, flow)
                warped_mask = self.warp_frame(self.prev_mask, flow)

                # Create initialization from warped previous frame
                # Blend current frame with warped previous in mask region
                mask_3ch = np.stack([current_mask] * 3, axis=-1) / 255.0

                # Initialize with blend of current and warped previous
                init_image = (
                    current_frame * (1 - mask_3ch * temporal_consistency) +
                    warped_prev * mask_3ch * temporal_consistency
                ).astype(np.uint8)

                init_pil = Image.fromarray(init_image)
            else:
                init_pil = pil_image

            # Default prompt
            if not prompt:
                prompt = "natural background, seamless fill, realistic lighting, temporal consistency"

            # Perform inpainting with temporal initialization
            result = self.model(
                prompt=prompt,
                image=init_pil,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=0.8,  # Don't completely overwrite initialization
                **kwargs
            ).images[0]

            result_np = np.array(result)

            # Store for next frame
            self.prev_frame = result_np.copy()
            self.prev_mask = current_mask.copy()

            return result_np

    def reset(self):
        """Reset temporal state."""
        self.prev_frame = None
        self.prev_mask = None


class TemporalOutpainting(BaseModel):
    """Temporal outpainting with frame-to-frame consistency."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize temporal outpainting model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "temporal_outpainting"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.prev_result = None
        self.load_model()

    def load_model(self) -> None:
        """Load temporal outpainting model."""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print(f"Loading temporal outpainting model...")

            model_id = "stabilityai/stable-diffusion-2-inpainting"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"Temporal outpainting loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for temporal outpainting. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load temporal outpainting model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        extend_pixels: int = 256,
        direction: str = "all",
        prompt: str = "",
        temporal_consistency: float = 0.5,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> np.ndarray:
        """Perform temporal outpainting with consistency.

        Args:
            image: Input image/frame
            extend_pixels: Number of pixels to extend
            direction: Direction to extend
            prompt: Text prompt for outpainting
            temporal_consistency: Strength of temporal consistency (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            **kwargs: Additional inference parameters

        Returns:
            Outpainted image as numpy array
        """
        from fusion_ai.models.outpainting import StableDiffusionOutpainting

        # Use standard outpainting model as base
        outpaint_model = StableDiffusionOutpainting(
            device=self.device,
            cache_dir=self.cache_dir,
            use_fp16=self.use_fp16
        )

        # Get canvas and mask for outpainting
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        orig_w, orig_h = pil_image.size

        # Calculate target size based on direction
        if direction == "all":
            target_w = orig_w + 2 * extend_pixels
            target_h = orig_h + 2 * extend_pixels
            position = "center"
        elif direction == "horizontal":
            target_w = orig_w + 2 * extend_pixels
            target_h = orig_h
            position = "center"
        elif direction == "vertical":
            target_w = orig_w
            target_h = orig_h + 2 * extend_pixels
            position = "center"
        else:
            # Use standard outpainting for other directions
            return outpaint_model.infer(
                image, extend_pixels, direction, prompt,
                num_inference_steps, guidance_scale, **kwargs
            )

        # Create canvas
        canvas = Image.new('RGB', (target_w, target_h), (127, 127, 127))

        if position == "center":
            x = (target_w - orig_w) // 2
            y = (target_h - orig_h) // 2

        # If we have previous result, use it for temporal consistency
        if self.prev_result is not None and temporal_consistency > 0:
            # Place previous result on canvas with slight transparency
            prev_pil = Image.fromarray(self.prev_result)
            if prev_pil.size == canvas.size:
                canvas = Image.blend(canvas, prev_pil, alpha=temporal_consistency * 0.5)

        # Place current frame
        canvas.paste(pil_image, (x, y))

        # Create mask
        mask = outpaint_model.create_outpaint_mask(
            (orig_w, orig_h), (target_w, target_h), position
        )

        # Default prompt for seamless extension
        if not prompt:
            prompt = "seamless extension, natural continuation, same style and lighting, temporal consistency"

        # Perform outpainting
        result = self.model(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        ).images[0]

        result_np = np.array(result)

        # Store for next frame
        self.prev_result = result_np.copy()

        return result_np

    def reset(self):
        """Reset temporal state."""
        self.prev_result = None


class WANInpainting(BaseModel):
    """WAN (Warp-Aware Neural) inpainting for video with advanced temporal consistency."""

    def __init__(
        self,
        method: str = "sd",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        lora_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize WAN inpainting model.

        Args:
            method: Inpainting method (lama/sd)
            device: Device to run inference on
            cache_dir: Directory to cache models
            lora_dir: Directory containing LoRA files
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.method = method
        self.use_fp16 = use_fp16
        self.lora_dir = lora_dir
        model_name = f"wan_inpainting_{method}"
        super().__init__(model_name, device, cache_dir, **kwargs)

        # Temporal state
        self.prev_frame = None
        self.prev_result = None
        self.prev_mask = None
        self.prev_flow = None

        self.load_model()

    def load_model(self) -> None:
        """Load WAN inpainting model."""
        try:
            if self.method == "sd":
                from diffusers import StableDiffusionInpaintPipeline
                from fusion_ai.utils.lora import LoRAManager

                print(f"Loading WAN inpainting (Stable Diffusion)...")

                model_id = "stabilityai/stable-diffusion-2-inpainting"
                torch_dtype = torch.float16 if self.use_fp16 else torch.float32

                self.model = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    cache_dir=str(self.cache_dir)
                )

                self.model = self.model.to(self.device)
                self.lora_manager = LoRAManager(self.lora_dir)

            elif self.method == "lama":
                from simple_lama_inpainting import SimpleLama

                print(f"Loading WAN inpainting (LaMa)...")
                self.model = SimpleLama()

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"WAN inpainting loaded successfully on {self.device} ({precision})")

        except ImportError as e:
            raise ImportError(
                f"Dependencies required for WAN inpainting ({self.method}). "
                f"Install with: pip install diffusers simple-lama-inpainting"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load WAN inpainting model: {e}")

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        method: str = "farneback"
    ) -> Optional[np.ndarray]:
        """Compute optical flow between frames using advanced methods.

        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)
            method: Flow method (farneback/raft)

        Returns:
            Optical flow field or None if unavailable
        """
        try:
            import cv2

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            if method == "farneback":
                # Dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5, levels=5, winsize=21,
                    iterations=5, poly_n=7, poly_sigma=1.5,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                )
            else:
                # Fallback to basic method
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

            return flow

        except ImportError:
            return None

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        occlusion_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp frame using optical flow with occlusion detection.

        Args:
            frame: Frame to warp (RGB)
            flow: Optical flow field
            occlusion_mask: Optional occlusion mask

        Returns:
            Tuple of (warped frame, confidence map)
        """
        try:
            import cv2

            h, w = flow.shape[:2]

            # Create coordinate grids
            flow_map = np.copy(flow)
            flow_map[:, :, 0] += np.arange(w)
            flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

            # Warp frame
            warped = cv2.remap(
                frame,
                flow_map.astype(np.float32),
                None,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Compute confidence map
            # Areas with large flow magnitude have lower confidence
            flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            confidence = np.exp(-flow_magnitude / 10.0)  # Exponential decay

            # Handle occlusions
            if occlusion_mask is not None:
                confidence *= (1.0 - occlusion_mask)

            # Expand confidence to 3 channels
            confidence_3ch = np.stack([confidence] * 3, axis=-1)

            return warped, confidence_3ch

        except Exception:
            # Fallback: return original frame with low confidence
            return frame, np.ones_like(frame) * 0.1

    def detect_occlusions(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        threshold: float = 1.0
    ) -> np.ndarray:
        """Detect occluded regions using forward-backward consistency.

        Args:
            forward_flow: Forward optical flow (t -> t+1)
            backward_flow: Backward optical flow (t+1 -> t)
            threshold: Occlusion threshold

        Returns:
            Binary occlusion mask (1 = occluded, 0 = visible)
        """
        try:
            import cv2

            h, w = forward_flow.shape[:2]

            # Warp backward flow using forward flow
            flow_map = forward_flow.copy()
            flow_map[:, :, 0] += np.arange(w)
            flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

            warped_backward = cv2.remap(
                backward_flow,
                flow_map.astype(np.float32),
                None,
                cv2.INTER_LINEAR
            )

            # Compute consistency error
            consistency_error = np.sqrt(
                np.sum((forward_flow + warped_backward) ** 2, axis=2)
            )

            # Threshold to get occlusion mask
            occlusion_mask = (consistency_error > threshold).astype(np.float32)

            return occlusion_mask

        except Exception:
            # Fallback: no occlusions detected
            return np.zeros(forward_flow.shape[:2], dtype=np.float32)

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "",
        negative_prompt: str = "blurry, bad quality, distorted",
        temporal_consistency: float = 0.85,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        detect_occlusions: bool = True,
        lora_paths: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> np.ndarray:
        """Perform WAN inpainting with advanced temporal consistency.

        Args:
            image: Input image
            mask: Binary mask (white = remove, black = keep)
            prompt: Text prompt (for SD method)
            negative_prompt: Negative prompt
            temporal_consistency: Consistency strength (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            detect_occlusions: Whether to detect and handle occlusions
            lora_paths: Optional LoRA file path(s)
            lora_weights: Optional LoRA weight(s)
            **kwargs: Additional parameters

        Returns:
            Inpainted image as numpy array
        """
        # Preprocess inputs
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        pil_mask = self.preprocess_image(mask, preserve_alpha=False)

        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')

        current_frame = np.array(pil_image)
        current_mask = np.array(pil_mask)

        # Initialize result
        result_np = None

        # Use temporal consistency if we have previous frame
        if self.prev_frame is not None and temporal_consistency > 0:
            # Compute forward and backward flow
            forward_flow = self.compute_optical_flow(self.prev_frame, current_frame)
            backward_flow = self.compute_optical_flow(current_frame, self.prev_frame)

            if forward_flow is not None and backward_flow is not None:
                # Detect occlusions
                occlusion_mask = None
                if detect_occlusions:
                    occlusion_mask = self.detect_occlusions(
                        forward_flow,
                        backward_flow,
                        threshold=1.0
                    )

                # Warp previous result
                if self.prev_result is not None:
                    warped_prev, confidence = self.warp_frame(
                        self.prev_result,
                        forward_flow,
                        occlusion_mask
                    )

                    # Create warp-aware initialization
                    mask_3ch = np.stack([current_mask] * 3, axis=-1) / 255.0

                    # Blend based on confidence and temporal consistency
                    blend_weight = confidence * mask_3ch * temporal_consistency

                    init_image = (
                        current_frame * (1 - blend_weight) +
                        warped_prev * blend_weight
                    ).astype(np.uint8)

                    init_pil = Image.fromarray(init_image)
                else:
                    init_pil = pil_image
            else:
                init_pil = pil_image
        else:
            init_pil = pil_image

        # Perform inpainting based on method
        if self.method == "sd":
            # Apply LoRAs if provided
            if lora_paths:
                if isinstance(lora_paths, str):
                    lora_paths = [lora_paths]
                if lora_weights is None:
                    lora_weights = [0.8] * len(lora_paths)
                elif isinstance(lora_weights, (int, float)):
                    lora_weights = [lora_weights] * len(lora_paths)

                for lora_path, lora_weight in zip(lora_paths, lora_weights):
                    self.model = self.lora_manager.load_lora(
                        self.model,
                        lora_path,
                        weight=lora_weight
                    )

            # Default prompt
            if not prompt:
                prompt = "natural background, seamless fill, realistic lighting, temporal consistency"

            # Stable Diffusion inpainting
            result = self.model(
                prompt=prompt,
                image=init_pil,
                mask_image=pil_mask,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=0.9,
                **kwargs
            ).images[0]

            # Unload LoRAs
            if lora_paths:
                self.lora_manager.unload_loras(self.model)

            result_np = np.array(result)

        elif self.method == "lama":
            # LaMa inpainting
            result = self.model(init_pil, pil_mask)
            result_np = np.array(result) if isinstance(result, Image.Image) else result

        # Store state for next frame
        self.prev_frame = current_frame.copy()
        self.prev_result = result_np.copy()
        self.prev_mask = current_mask.copy()

        return result_np

    def reset(self):
        """Reset temporal state."""
        self.prev_frame = None
        self.prev_result = None
        self.prev_mask = None
        self.prev_flow = None
