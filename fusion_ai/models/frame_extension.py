"""Frame extension for creating extra frames at shot beginning/end."""

from pathlib import Path
from typing import Optional, Union, List
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class FrameExtension(BaseModel):
    """Frame extension using temporal inpainting and motion prediction."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize frame extension model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "frame_extension"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load frame interpolation and inpainting models."""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print(f"Loading frame extension models...")

            # Load SD inpainting for temporal inpainting
            model_id = "stabilityai/stable-diffusion-2-inpainting"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.inpaint_model = self.inpaint_model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"Frame extension loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for frame extension. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load frame extension model: {e}")

    def estimate_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """Estimate motion between two frames using optical flow.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Optical flow field
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
            print("OpenCV not available for motion estimation, using simple prediction")
            return None

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        scale: float = 1.0
    ) -> np.ndarray:
        """Warp frame using optical flow.

        Args:
            frame: Frame to warp
            flow: Optical flow field
            scale: Scale factor for flow (1.0 = one frame forward)

        Returns:
            Warped frame
        """
        import cv2

        h, w = frame.shape[:2]
        flow_scaled = flow * scale

        # Create coordinate maps
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x += flow_scaled[:, :, 0]
        y += flow_scaled[:, :, 1]

        # Remap
        warped = cv2.remap(frame, x, y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

        return warped

    @torch.no_grad()
    def infer(
        self,
        frames: List[Union[str, Path, Image.Image, np.ndarray]],
        direction: str = "forward",
        num_new_frames: int = 5,
        blend_mode: str = "optical_flow",
        use_inpainting: bool = True,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate extended frames at beginning or end of sequence.

        Args:
            frames: List of 2-5 reference frames from shot
            direction: "forward" (extend end) or "backward" (extend beginning)
            num_new_frames: Number of new frames to generate
            blend_mode: "optical_flow", "simple", or "inpainting"
            use_inpainting: Use AI inpainting to clean up artifacts
            **kwargs: Additional parameters

        Returns:
            List of generated frames
        """
        # Preprocess frames
        frame_arrays = []
        for frame in frames:
            pil_frame = self.preprocess_image(frame, preserve_alpha=False)
            frame_arrays.append(np.array(pil_frame))

        if len(frame_arrays) < 2:
            raise ValueError("Need at least 2 reference frames for extension")

        # Generate new frames
        new_frames = []

        if blend_mode == "optical_flow" and len(frame_arrays) >= 2:
            # Use optical flow for motion estimation
            if direction == "forward":
                ref_frames = frame_arrays[-2:]
            else:
                ref_frames = frame_arrays[:2]

            # Estimate motion
            flow = self.estimate_motion(ref_frames[0], ref_frames[1])

            if flow is not None:
                # Generate frames by warping
                for i in range(1, num_new_frames + 1):
                    if direction == "forward":
                        warped = self.warp_frame(ref_frames[1], flow, scale=i)
                    else:
                        warped = self.warp_frame(ref_frames[0], flow, scale=-i)

                    new_frames.append(warped)

            else:
                # Fallback to simple blending
                blend_mode = "simple"

        if blend_mode == "simple" or not new_frames:
            # Simple frame blending/extrapolation
            if direction == "forward":
                base_frame = frame_arrays[-1]
            else:
                base_frame = frame_arrays[0]

            # Generate frames with slight variations
            for i in range(num_new_frames):
                # For now, just duplicate with slight blending
                if len(frame_arrays) >= 2:
                    if direction == "forward":
                        alpha = 0.9 ** (i + 1)
                        blended = (base_frame * alpha +
                                 frame_arrays[-2] * (1 - alpha)).astype(np.uint8)
                    else:
                        alpha = 0.9 ** (i + 1)
                        blended = (base_frame * alpha +
                                 frame_arrays[1] * (1 - alpha)).astype(np.uint8)

                    new_frames.append(blended)
                else:
                    new_frames.append(base_frame.copy())

        # Optional: Use inpainting to clean up generated frames
        if use_inpainting and hasattr(self, 'inpaint_model'):
            refined_frames = []
            for frame in new_frames:
                # Create edge mask to inpaint uncertain regions
                # For now, skip inpainting to keep it fast
                refined_frames.append(frame)
            new_frames = refined_frames

        return new_frames


class FrameInterpolation(BaseModel):
    """Frame interpolation for smooth slow-motion and frame rate conversion."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize frame interpolation model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        model_name = "frame_interpolation"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load frame interpolation model (RIFE, FILM, etc.)."""
        print("Frame interpolation model initialized")
        # Placeholder for RIFE or FILM integration
        # These would provide high-quality frame interpolation
        self.model_loaded = True

    @torch.no_grad()
    def infer(
        self,
        frame1: Union[str, Path, Image.Image, np.ndarray],
        frame2: Union[str, Path, Image.Image, np.ndarray],
        num_interpolated: int = 1,
        **kwargs
    ) -> List[np.ndarray]:
        """Interpolate frames between two frames.

        Args:
            frame1: First frame
            frame2: Second frame
            num_interpolated: Number of frames to interpolate between
            **kwargs: Additional parameters

        Returns:
            List of interpolated frames
        """
        # Preprocess frames
        pil_frame1 = self.preprocess_image(frame1, preserve_alpha=False)
        pil_frame2 = self.preprocess_image(frame2, preserve_alpha=False)

        frame1_np = np.array(pil_frame1)
        frame2_np = np.array(pil_frame2)

        # Simple linear interpolation for now
        # Can be replaced with RIFE or FILM for better quality
        interpolated = []
        for i in range(1, num_interpolated + 1):
            t = i / (num_interpolated + 1)
            blended = ((1 - t) * frame1_np + t * frame2_np).astype(np.uint8)
            interpolated.append(blended)

        return interpolated
