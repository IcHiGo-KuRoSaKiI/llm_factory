
# examples/lmstudio_vision_example.py
import os
import json
import logging
from typing import Optional
import argparse

# Import the llm_factory components
from llm_factory import run_pipeline, LLMClientFactory
from utils.image_utils import image_to_data_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image_with_lmstudio(
    image_path: str,
    base_url: str = "http://localhost:1234",
    custom_prompt: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Process an image with LM Studio running a vision-enabled model locally
    
    Args:
        image_path: Path to the image file
        base_url: URL of the LM Studio server
        custom_prompt: Optional custom prompt to use for image analysis
        output_path: Optional path to save the results
        
    Returns:
        Extracted text from the image
    """
    try:
        # Create the LM Studio client with vision support
        client_params = {
            "base_url": base_url,
            "supports_vision": True  # Explicitly state that model supports vision
        }
        
        # Convert image to data URI
        logger.info(f"Converting image to base64: {image_path}")
        image_data_uri = image_to_data_uri(image_path)
        
        if not image_data_uri:
            raise ValueError(f"Failed to encode image: {image_path}")
        
        # Create the client directly
        logger.info("Creating LM Studio client for image processing")
        client = LLMClientFactory.create_client(client_type="lmstudio", **client_params)
        
        # Process the image
        logger.info("Processing image with LM Studio client")
        extracted_text = client.get_openai_response_image(
            image_data=image_data_uri,
            prompt=custom_prompt
        )
        
        # Save to file if output path provided
        if output_path:
            logger.info(f"Saving results to {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                result = {
                    "client_type": "lmstudio",
                    "image_path": image_path,
                    "extracted_text": extracted_text
                }
                json.dump(result, f, indent=2)
        
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return f"[Image processing failed: {str(e)}]"

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Process images using LM Studio with vision capabilities")
    parser.add_argument("--image", required=True, help="Path to the image file to process")
    parser.add_argument("--server", default="http://localhost:1234", help="LM Studio server URL")
    parser.add_argument("--output", help="Path to save the output to a JSON file")
    args = parser.parse_args()
    
    # Define a custom prompt
    custom_prompt = """
    You are an AI assistant specialized in analyzing business documents and diagrams. 
    Please extract all text content from this image, preserving formatting where relevant.
    Pay special attention to any tables, charts, or diagrams, and describe their structure and content.
    Ignore any watermarks or background elements that aren't part of the main content.
    """
    
    # Process the image
    result = process_image_with_lmstudio(
        image_path=args.image,
        base_url=args.server,
        custom_prompt=custom_prompt,
        output_path=args.output
    )
    
    # Print the result
    print("\n=== LM Studio Vision Result ===")
    print(result[:500] + "..." if len(result) > 500 else result)