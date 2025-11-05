"""ControlNet models for conditional image generation."""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import torch
import numpy as np
from PIL import Image
import cv2

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class ControlNetPreprocessor(BaseModel):
    """Base class for ControlNet preprocessors."""

    def __init__(
        self,
        preprocessor_type: str,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize ControlNet preprocessor.

        Args:
            preprocessor_type: Type of preprocessor (openpose, canny, depth, etc.)
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        self.preprocessor_type = preprocessor_type
        model_name = f"controlnet_{preprocessor_type}"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the preprocessor model."""
        raise NotImplementedError

    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for ControlNet.

        Args:
            image: Input image

        Returns:
            Preprocessed control image
        """
        raise NotImplementedError


class OpenPoseDetector(ControlNetPreprocessor):
    """OpenPose detector for human pose estimation."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize OpenPose detector.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        super().__init__("openpose", device, cache_dir, **kwargs)

    def load_model(self) -> None:
        """Load OpenPose model."""
        try:
            from controlnet_aux import OpenposeDetector

            print(f"Loading OpenPose detector...")
            self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            print(f"OpenPose detector loaded successfully on {self.device}")

        except ImportError:
            print("Warning: controlnet_aux not installed. Attempting fallback...")
            try:
                # Fallback to mediapipe for pose detection
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                self.use_mediapipe = True
                print(f"Using MediaPipe fallback for pose detection on {self.device}")
            except ImportError:
                raise ImportError(
                    "Neither controlnet_aux nor mediapipe available. "
                    "Install with: pip install controlnet-aux or pip install mediapipe"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenPose detector: {e}")

    @torch.no_grad()
    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Detect pose keypoints in image.

        Args:
            image: Input image

        Returns:
            OpenPose skeleton visualization as numpy array
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, preserve_alpha=False)

        if hasattr(self, 'use_mediapipe') and self.use_mediapipe:
            # Use MediaPipe fallback
            img_array = np.array(pil_image)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results = self.detector.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

            # Create black canvas
            pose_image = np.zeros_like(img_array)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    pose_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

            return pose_image
        else:
            # Use controlnet_aux
            pose_image = self.detector(pil_image)
            return np.array(pose_image)

    def detect_pose_sequence(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]]
    ) -> List[np.ndarray]:
        """Detect poses in a sequence of images.

        Args:
            images: List of input images

        Returns:
            List of pose visualizations
        """
        pose_sequence = []
        for img in images:
            pose = self.preprocess(img)
            pose_sequence.append(pose)
        return pose_sequence


class CannyDetector(ControlNetPreprocessor):
    """Canny edge detector for edge-based control."""

    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize Canny detector.

        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Higher threshold for edge detection
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        super().__init__("canny", device, cache_dir, **kwargs)

    def load_model(self) -> None:
        """Load Canny detector (OpenCV-based, no model loading needed)."""
        print(f"Canny edge detector initialized")

    @torch.no_grad()
    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Detect edges in image using Canny.

        Args:
            image: Input image

        Returns:
            Edge map as numpy array
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        img_array = np.array(pil_image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        # Convert back to RGB (3 channels)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_rgb


class ControlNetSD(BaseModel):
    """Stable Diffusion with ControlNet conditioning."""

    def __init__(
        self,
        controlnet_type: str = "openpose",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        lora_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize ControlNet Stable Diffusion.

        Args:
            controlnet_type: Type of ControlNet (openpose, canny, depth, etc.)
            device: Device to run inference on
            cache_dir: Directory to cache models
            lora_dir: Directory containing LoRA files
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.controlnet_type = controlnet_type
        self.use_fp16 = use_fp16
        self.lora_dir = lora_dir
        model_name = f"controlnet_sd_{controlnet_type}"
        super().__init__(model_name, device, cache_dir, **kwargs)

        # Load preprocessor
        if controlnet_type == "openpose":
            self.preprocessor = OpenPoseDetector(device=device, cache_dir=cache_dir)
        elif controlnet_type == "canny":
            self.preprocessor = CannyDetector(device=device, cache_dir=cache_dir)
        else:
            self.preprocessor = None

        self.load_model()

    def load_model(self) -> None:
        """Load ControlNet and Stable Diffusion pipeline."""
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler
            )
            from fusion_ai.utils.lora import LoRAManager

            print(f"Loading ControlNet ({self.controlnet_type})...")

            # Load ControlNet model
            controlnet_model_map = {
                "openpose": "lllyasviel/control_v11p_sd15_openpose",
                "canny": "lllyasviel/control_v11p_sd15_canny",
                "depth": "lllyasviel/control_v11f1p_sd15_depth",
                "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
                "normal": "lllyasviel/control_v11p_sd15_normalbae",
                "scribble": "lllyasviel/control_v11p_sd15_scribble",
                "seg": "lllyasviel/control_v11p_sd15_seg",
            }

            controlnet_id = controlnet_model_map.get(
                self.controlnet_type,
                "lllyasviel/control_v11p_sd15_openpose"
            )

            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            # Load Stable Diffusion pipeline with ControlNet
            base_model_id = "runwayml/stable-diffusion-v1-5"
            self.model = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir),
                safety_checker=None
            )

            # Use faster scheduler
            self.model.scheduler = UniPCMultistepScheduler.from_config(
                self.model.scheduler.config
            )

            self.model = self.model.to(self.device)

            # Enable memory optimizations
            try:
                self.model.enable_attention_slicing()
                self.model.enable_vae_slicing()
            except:
                pass

            # Initialize LoRA manager
            self.lora_manager = LoRAManager(self.lora_dir)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"ControlNet SD loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for ControlNet. "
                "Install it with: pip install diffusers controlnet-aux"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ControlNet: {e}")

    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        control_image: Union[str, Path, Image.Image, np.ndarray],
        negative_prompt: str = "low quality, blurry, distorted",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        lora_paths: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate image conditioned on control image.

        Args:
            prompt: Text prompt
            control_image: Control image (can be raw image, will be preprocessed)
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            controlnet_conditioning_scale: ControlNet influence strength (0-2)
            seed: Random seed
            width: Output width (must be multiple of 8)
            height: Output height (must be multiple of 8)
            lora_paths: Optional LoRA file path(s)
            lora_weights: Optional LoRA weight(s)
            **kwargs: Additional parameters

        Returns:
            Generated image as numpy array
        """
        # Preprocess control image
        control_pil = self.preprocess_image(control_image, preserve_alpha=False)

        # Determine output size
        if width is None or height is None:
            width, height = control_pil.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        control_pil = control_pil.resize((width, height), Image.LANCZOS)

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

        # Set seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate image
        result = self.model(
            prompt=prompt,
            image=control_pil,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            width=width,
            height=height,
            **kwargs
        )

        # Unload LoRAs
        if lora_paths:
            self.lora_manager.unload_loras(self.model)

        # Convert to numpy
        result_image = result.images[0]
        return np.array(result_image)

    def generate_from_pose_sequence(
        self,
        prompt: str,
        pose_images: List[Union[str, Path, Image.Image, np.ndarray]],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate video frames from pose sequence.

        Args:
            prompt: Text prompt
            pose_images: List of pose control images
            **kwargs: Additional parameters for infer()

        Returns:
            List of generated frames
        """
        frames = []
        for pose_img in pose_images:
            frame = self.infer(prompt, pose_img, **kwargs)
            frames.append(frame)
        return frames
