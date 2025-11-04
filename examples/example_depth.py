"""Example: Generate depth maps using DepthAnythingV2."""

from pathlib import Path
from fusion_ai.models import DepthAnythingV2
from fusion_ai.utils import save_image


def example_basic_depth():
    """Basic depth map generation example."""
    print("Example 1: Basic Depth Map Generation")
    print("=" * 50)

    # Initialize model (will download on first use)
    print("Loading DepthAnythingV2 base model...")
    model = DepthAnythingV2(model_size="base", device="cuda")

    # Generate depth map from image
    print("Generating depth map...")
    depth_map = model.infer(
        "path/to/your/image.jpg",
        normalize=True
    )

    # Save output
    save_image(depth_map, "output_depth.png")
    print("Depth map saved to: output_depth.png")


def example_depth_with_colormap():
    """Generate colored depth map example."""
    print("\nExample 2: Depth Map with Colormap")
    print("=" * 50)

    model = DepthAnythingV2(model_size="base", device="cuda")

    # Generate depth map with colormap
    print("Generating colored depth map...")
    depth_map = model.infer(
        "path/to/your/image.jpg",
        normalize=True,
        colormap="COLORMAP_INFERNO"  # Options: INFERNO, VIRIDIS, PLASMA, etc.
    )

    save_image(depth_map, "output_depth_colored.png")
    print("Colored depth map saved to: output_depth_colored.png")


def example_batch_processing():
    """Batch process multiple images."""
    print("\nExample 3: Batch Processing")
    print("=" * 50)

    model = DepthAnythingV2(model_size="small", device="cuda")  # Use small model for speed

    # List of images to process
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]

    print(f"Processing {len(image_paths)} images...")
    depth_maps = model.infer_batch(image_paths, normalize=True)

    # Save all outputs
    for i, depth_map in enumerate(depth_maps):
        output_path = f"output_depth_batch_{i}.png"
        save_image(depth_map, output_path)
        print(f"  Saved: {output_path}")


def example_model_comparison():
    """Compare different model sizes."""
    print("\nExample 4: Model Size Comparison")
    print("=" * 50)

    image_path = "path/to/your/image.jpg"

    for size in ["small", "base", "large"]:
        print(f"\nProcessing with {size} model...")
        model = DepthAnythingV2(model_size=size, device="cuda")

        depth_map = model.infer(image_path, normalize=True)
        save_image(depth_map, f"output_depth_{size}.png")
        print(f"  Saved: output_depth_{size}.png")


def example_high_precision():
    """Generate high-precision depth map (EXR format)."""
    print("\nExample 5: High Precision Depth (EXR)")
    print("=" * 50)

    model = DepthAnythingV2(model_size="large", device="cuda")

    # Generate without normalization for full precision
    depth_map = model.infer(
        "path/to/your/image.jpg",
        normalize=False  # Keep original depth values
    )

    # Save as EXR for 32-bit precision
    model.save_depth(depth_map, "output_depth.exr", format="exr")
    print("High-precision depth saved to: output_depth.exr")


if __name__ == "__main__":
    # Run examples
    # Note: Replace "path/to/your/image.jpg" with actual image paths

    print("DepthAnythingV2 Examples")
    print("=" * 50)
    print("\nNote: These examples require actual image files.")
    print("Replace 'path/to/your/image.jpg' with your image paths.\n")

    # Uncomment the examples you want to run:

    # example_basic_depth()
    # example_depth_with_colormap()
    # example_batch_processing()
    # example_model_comparison()
    # example_high_precision()

    print("\nExamples complete!")
