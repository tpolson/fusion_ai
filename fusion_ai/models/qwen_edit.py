"""Qwen Edit model implementation for image editing."""

from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import MODEL_CONFIGS, CACHE_DIR


class QwenEdit(BaseModel):
    """Qwen VL model for text-guided image editing."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_flash_attn: bool = False,
        **kwargs
    ):
        """Initialize Qwen Edit model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_flash_attn: Whether to use Flash Attention
            **kwargs: Additional arguments
        """
        self.use_flash_attn = use_flash_attn
        model_name = "qwen_vl_chat"
        super().__init__(model_name, device, cache_dir, **kwargs)

        self.config = MODEL_CONFIGS["qwen_edit"]["default"]
        self.load_model()

    def load_model(self) -> None:
        """Load the Qwen VL model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading Qwen VL Chat model...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["repo_id"],
                trust_remote_code=True,
                cache_dir=str(self.cache_dir)
            )

            # Load model
            model_kwargs = {
                "device_map": self.device if self.device == "cpu" else "auto",
                "trust_remote_code": True,
                "cache_dir": str(self.cache_dir),
            }

            if self.use_flash_attn:
                model_kwargs["use_flash_attn"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["repo_id"],
                **model_kwargs
            )

            self.model.eval()

            print(f"Qwen VL Chat loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers is required for QwenEdit. "
                "Install it with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen VL model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Perform visual question answering or image understanding.

        Args:
            image: Input image
            prompt: Text prompt for the model
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Model response text
        """
        # Preprocess image
        pil_image = self.preprocess_image(image)

        # Save image temporarily for Qwen input format
        temp_image_path = self.cache_dir / "temp_input.png"
        pil_image.save(temp_image_path)

        # Prepare query with image
        query = self.tokenizer.from_list_format([
            {'image': str(temp_image_path)},
            {'text': prompt},
        ])

        # Generate response
        response, history = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

        return response

    def edit(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        instruction: str,
        **kwargs
    ) -> str:
        """Edit an image based on text instruction.

        Args:
            image: Input image
            instruction: Editing instruction
            **kwargs: Additional parameters

        Returns:
            Description of the edit or instructions for editing
        """
        prompt = f"Please describe how to edit this image: {instruction}"
        return self.infer(image, prompt, **kwargs)

    def describe(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
    ) -> str:
        """Generate a description of the image.

        Args:
            image: Input image
            **kwargs: Additional parameters

        Returns:
            Image description
        """
        prompt = "Describe this image in detail."
        return self.infer(image, prompt, **kwargs)

    def answer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        question: str,
        **kwargs
    ) -> str:
        """Answer a question about the image.

        Args:
            image: Input image
            question: Question to answer
            **kwargs: Additional parameters

        Returns:
            Answer to the question
        """
        return self.infer(image, question, **kwargs)
