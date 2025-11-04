"""Example: Image understanding using Qwen VL."""

from fusion_ai.models import QwenEdit


def example_describe_image():
    """Describe an image in detail."""
    print("Example 1: Image Description")
    print("=" * 50)

    # Initialize model (will download on first use)
    print("Loading Qwen VL Chat model...")
    model = QwenEdit(device="cuda")

    # Describe the image
    print("Generating description...")
    description = model.describe("path/to/your/image.jpg")

    print("\nDescription:")
    print(description)


def example_answer_question():
    """Answer questions about an image."""
    print("\nExample 2: Visual Question Answering")
    print("=" * 50)

    model = QwenEdit(device="cuda")

    # Ask questions about the image
    questions = [
        "What objects are in this image?",
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = model.answer("path/to/your/image.jpg", question)
        print(f"Answer: {answer}")


def example_edit_instruction():
    """Get edit instructions for an image."""
    print("\nExample 3: Image Edit Instructions")
    print("=" * 50)

    model = QwenEdit(device="cuda")

    # Get editing instructions
    instructions = [
        "Make the sky more dramatic",
        "Add warmer tones",
        "Increase contrast",
    ]

    for instruction in instructions:
        print(f"\nInstruction: {instruction}")
        result = model.edit("path/to/your/image.jpg", instruction)
        print(f"Result: {result}")


def example_custom_prompt():
    """Use custom prompts for specific tasks."""
    print("\nExample 4: Custom Prompts")
    print("=" * 50)

    model = QwenEdit(device="cuda")

    # Custom prompts
    prompts = [
        "List all the people and objects visible in this image",
        "Describe the composition and lighting of this photo",
        "What emotion or mood does this image convey?",
        "Identify any text visible in this image",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = model.infer("path/to/your/image.jpg", prompt)
        print(f"Response: {result}")


def example_image_comparison():
    """Compare multiple images."""
    print("\nExample 5: Image Analysis")
    print("=" * 50)

    model = QwenEdit(device="cuda")

    images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]

    prompt = "Describe the style and content of this image in one sentence"

    for i, image_path in enumerate(images, 1):
        print(f"\nImage {i}:")
        description = model.infer(image_path, prompt)
        print(f"  {description}")


def example_detailed_analysis():
    """Perform detailed image analysis."""
    print("\nExample 6: Detailed Analysis")
    print("=" * 50)

    model = QwenEdit(device="cuda")

    image_path = "path/to/your/image.jpg"

    # Multiple aspects of analysis
    analyses = {
        "Content": "What is the main subject and what else is visible?",
        "Technical": "Describe the photography technique, lighting, and composition",
        "Artistic": "What artistic style or influences are apparent?",
        "Context": "What might be the context or story behind this image?",
    }

    print(f"Analyzing: {image_path}\n")

    for aspect, question in analyses.items():
        print(f"{aspect} Analysis:")
        result = model.infer(image_path, question)
        print(f"  {result}\n")


if __name__ == "__main__":
    # Run examples
    # Note: Replace "path/to/your/image.jpg" with actual image paths

    print("Qwen VL Examples")
    print("=" * 50)
    print("\nNote: These examples require actual image files.")
    print("Replace 'path/to/your/image.jpg' with your image paths.\n")
    print("WARNING: First run will download the Qwen VL model (~10GB)")
    print("=" * 50)

    # Uncomment the examples you want to run:

    # example_describe_image()
    # example_answer_question()
    # example_edit_instruction()
    # example_custom_prompt()
    # example_image_comparison()
    # example_detailed_analysis()

    print("\nExamples complete!")
