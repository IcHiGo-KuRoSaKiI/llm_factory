"""Example demonstrating processor-level routing between OpenAI and local vision models."""
import os
from llm_factory.processors.standard_processor import StandardPromptProcessor
from llm_factory.utils import image_to_data_uri


def main() -> None:
    processor = StandardPromptProcessor()

    # Text-based call -> always OpenAI
    text_result = processor.get_completion("Explain the moon landing in one sentence.")
    print("Text Response:\n", text_result)

    # Vision-based call -> routed to LM Studio or Ollama depending on ENV
    sample_image = os.path.join(os.path.dirname(__file__), "sample.jpg")
    if os.path.isfile(sample_image):
        data_uri = image_to_data_uri(sample_image)
        image_result = processor.get_response_image("What is in this picture?", data_uri)
        print("Vision Response:\n", image_result)
    else:
        print("Sample image not found; skipping vision request")


if __name__ == "__main__":
    main()
