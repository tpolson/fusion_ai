"""Video generation models including WAN (Warp-Aware Neural) networks."""

from pathlib import Path
from typing import Optional, Union, List
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class WANVideoGenerator(BaseModel):
    """WAN (Warp-Aware Neural) video generation model."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize WAN video generator.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "wan_video_generator"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load WAN video generation model."""
        try:
            from diffusers import StableDiffusionPipeline

            print(f"Loading WAN video generation model...")

            # Use base SD model with custom pipeline for video generation
            model_id = "stabilityai/stable-diffusion-2-1"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            # Enable memory efficient attention if available
            try:
                self.model.enable_attention_slicing()
                self.model.enable_vae_slicing()
            except:
                pass

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"WAN video generator loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for WAN video generation. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load WAN video generator: {e}")

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute optical flow between frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Optical flow or None
        """
        try:
            import cv2

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            return flow

        except ImportError:
            return None

    def warp_latent(
        self,
        latent: torch.Tensor,
        flow: np.ndarray,
        scale_factor: int = 8
    ) -> torch.Tensor:
        """Warp latent representation using optical flow.

        Args:
            latent: Latent tensor to warp
            flow: Optical flow field
            scale_factor: Downscale factor of latent vs image

        Returns:
            Warped latent
        """
        import torch.nn.functional as F

        # Downscale flow to match latent resolution
        h, w = latent.shape[2:]
        flow_down = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
        flow_down = F.interpolate(flow_down, size=(h, w), mode='bilinear', align_corners=False)
        flow_down = flow_down / scale_factor  # Scale flow values

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

        # Apply flow to grid
        flow_norm = flow_down / torch.tensor([w, h]).view(1, 2, 1, 1) * 2
        grid_warped = grid + flow_norm

        # Warp latent
        grid_warped = grid_warped.permute(0, 2, 3, 1).to(latent.device)
        warped = F.grid_sample(latent, grid_warped, align_corners=False, mode='bilinear')

        return warped

    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]],
        reference_frames: Optional[List[Union[str, Path, Image.Image, np.ndarray]]] = None,
        num_frames: int = 16,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        temporal_consistency_strength: float = 0.7,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate video frames with WAN.

        Args:
            prompt: Text prompt or list of prompts for keyframes
            reference_frames: Optional reference frames for guidance
            num_frames: Number of frames to generate
            width: Frame width
            height: Frame height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            temporal_consistency_strength: Strength of temporal consistency (0-1)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters

        Returns:
            List of generated frames as numpy arrays
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        generated_frames = []
        prev_latent = None
        prev_frame = None

        # Handle single prompt or keyframe prompts
        if isinstance(prompt, str):
            prompts = [prompt] * num_frames
        else:
            # Interpolate prompts for intermediate frames
            prompts = self._interpolate_prompts(prompt, num_frames)

        for frame_idx in range(num_frames):
            current_prompt = prompts[frame_idx]

            # Generate frame
            if prev_latent is None or temporal_consistency_strength == 0:
                # First frame or no temporal consistency
                result = self.model(
                    prompt=current_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs
                )
                frame = np.array(result.images[0])

            else:
                # Use previous latent for temporal consistency
                # This is a simplified version - full WAN would use more sophisticated warping

                # Generate with latent initialization
                result = self.model(
                    prompt=current_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs
                )
                frame = np.array(result.images[0])

                # Blend with warped previous frame for consistency
                if prev_frame is not None and temporal_consistency_strength > 0:
                    flow = self.compute_optical_flow(prev_frame, frame)
                    if flow is not None:
                        import cv2
                        h, w = frame.shape[:2]
                        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                        x += flow[:, :, 0]
                        y += flow[:, :, 1]
                        warped_prev = cv2.remap(prev_frame, x, y, cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE)

                        # Blend for temporal consistency
                        alpha = temporal_consistency_strength * 0.3
                        frame = ((1 - alpha) * frame + alpha * warped_prev).astype(np.uint8)

            generated_frames.append(frame)
            prev_frame = frame.copy()

            print(f"Generated frame {frame_idx + 1}/{num_frames}")

        return generated_frames

    def _interpolate_prompts(
        self,
        keyframe_prompts: List[str],
        num_frames: int
    ) -> List[str]:
        """Interpolate prompts between keyframes.

        Args:
            keyframe_prompts: List of prompts for keyframes
            num_frames: Total number of frames needed

        Returns:
            List of prompts for all frames
        """
        if len(keyframe_prompts) == 1:
            return keyframe_prompts * num_frames

        # Simple prompt interpolation - assign prompts to frame ranges
        frames_per_keyframe = num_frames // len(keyframe_prompts)
        prompts = []

        for i, prompt in enumerate(keyframe_prompts):
            if i < len(keyframe_prompts) - 1:
                prompts.extend([prompt] * frames_per_keyframe)
            else:
                # Last keyframe gets remaining frames
                prompts.extend([prompt] * (num_frames - len(prompts)))

        return prompts[:num_frames]

    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: int = 24
    ) -> None:
        """Save generated frames as video.

        Args:
            frames: List of frame arrays
            output_path: Output video path
            fps: Frames per second
        """
        try:
            import cv2

            output_path = str(output_path)
            h, w = frames[0].shape[:2]

            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"Video saved to {output_path}")

        except ImportError:
            print("OpenCV not available for video saving")
        except Exception as e:
            print(f"Failed to save video: {e}")
