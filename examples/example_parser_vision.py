# How To RUN 
# python -m examples.example_parser_vision
import os
import json
import base64
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from parsers.factory import ParserFactory
from llm_factory import LLMClientFactory

# Load environment variables from .env file
load_dotenv()

# Mock logger for example purposes
class MockLogger:
    def log_success(self, file_path, schema_name, weaviate_uuid, operation_type):
        print(f"SUCCESS: {operation_type} operation on {file_path} successful.")
        
    def log_failure(self, file_path, error, error_type):
        print(f"FAILURE: {error_type} on {file_path}: {error}")

def process_image(
    image_path: str,
    client_type: str = "lmstudio",
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    **client_kwargs
) -> str:
    """
    Process a single image using the specified LLM client's vision capabilities.
    
    Args:
        image_path: Path to the image file
        client_type: Type of LLM client to use (openai, azure, groq, lmstudio, ollama)
        prompt: Optional custom prompt for image analysis
        output_path: Optional path to save the results as JSON
        **client_kwargs: Additional kwargs to pass to the client initialization
        
    Returns:
        Extracted text from the image
    """
    try:
        # Ensure the image file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # For LMStudio, ensure we set supports_vision to True
        if client_type.lower() == "lmstudio" and "supports_vision" not in client_kwargs:
            client_kwargs["supports_vision"] = True
            
        # Create the LLM client
        print(f"Creating {client_type} client for image processing...")
        client = LLMClientFactory.create_client(client_type=client_type, **client_kwargs)
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Convert to data URI if not already
        if not base64_encoded.startswith('data:'):
            # Determine the correct MIME type from file extension
            _, ext = os.path.splitext(image_path.lower())
            mime_type = "image/jpeg"  # default
            if ext == ".png":
                mime_type = "image/png"
            elif ext == ".gif":
                mime_type = "image/gif"
            elif ext == ".webp":
                mime_type = "image/webp"
                
            data_uri = f"data:{mime_type};base64,{base64_encoded}"
        else:
            data_uri = base64_encoded
            
        # Get custom prompt or use default
        if not prompt:
            prompt = """
            Extract all text content from this image. If the image contains diagrams, 
            charts, or tables, describe their structure and content. Ignore any 
            watermarks or background elements.
            """
            
        # Process the image
        print(f"Processing image using {client_type}...")
        extracted_text = client.get_openai_response_image(
            image_data=data_uri,
            prompt=prompt
        )
        
        # Save results if output path is provided
        if output_path:
            results = {
                "client_type": client_type,
                "image_path": image_path,
                "extracted_text": extracted_text
            }
            
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
            
        return extracted_text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"[Image processing failed: {str(e)}]"

def process_document(
    file_path: str,
    client_type: str = "lmstudio",
    output_path: Optional[str] = None,
    **client_kwargs
) -> List[Dict[str, Any]]:
    """
    Process a document using the parser factory with the specified LLM client
    for image extraction.
    
    Args:
        file_path: Path to the document file
        client_type: Type of LLM client to use for image processing
        output_path: Optional path to save the results as JSON
        **client_kwargs: Additional kwargs to pass to the client initialization
        
    Returns:
        Parsed document content
    """
    try:
        # Verify file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
            
        # For LMStudio, ensure we set supports_vision to True
        if client_type.lower() == "lmstudio" and "supports_vision" not in client_kwargs:
            client_kwargs["supports_vision"] = True
            
        # Create the LLM client
        print(f"Creating {client_type} client for document processing...")
        llm_client = LLMClientFactory.create_client(client_type=client_type, **client_kwargs)
        
        # Create logger
        logger = MockLogger()
        
        # Create parser using factory
        filename = os.path.basename(file_path)
        print(f"Processing {filename} using {client_type} for image extraction...")
        parser = ParserFactory.create_parser(file_path, llm_client, logger)
        
        # Parse the document
        result = parser.parse(file_path)
        
        # Print summary of extracted content
        print(f"Successfully extracted content from {filename}")
        print(f"Number of chunks/pages: {len(result)}")
        
        # Print short sample of the first chunk
        if result:
            sample_text = result[0].get("text", "")[:100]
            print(f"Sample text: {sample_text}...")
        
        # Save results to JSON file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(result, json_file, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
            
        return result
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

def process_document_directory(
    directory_path: str,
    client_type: str = "lmstudio",
    output_path: Optional[str] = None,
    **client_kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all supported documents in a directory using the parser factory.
    
    Args:
        directory_path: Path to the directory containing documents
        client_type: Type of LLM client to use for image processing
        output_path: Optional path to save results as JSON
        **client_kwargs: Additional kwargs to pass to the client initialization
        
    Returns:
        Dictionary with filenames as keys and parsed content as values
    """
    # For LMStudio, ensure we set supports_vision to True
    if client_type.lower() == "lmstudio" and "supports_vision" not in client_kwargs:
        client_kwargs["supports_vision"] = True
        
    # Create the LLM client
    print(f"Creating {client_type} client for document processing...")
    llm_client = LLMClientFactory.create_client(client_type=client_type, **client_kwargs)
    
    # Create logger
    logger = MockLogger()
    
    # Get list of supported extensions from factory
    supported_extensions = ParserFactory.get_supported_extensions()
    print(f"Supported file extensions: {', '.join(supported_extensions)}")
    
    # Dictionary to store all results
    all_results = {}
    
    # Process each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        try:
            # Check if the file extension is supported
            _, extension = os.path.splitext(filename.lower())
            if extension not in supported_extensions:
                print(f"Skipping {filename}: Unsupported file type")
                continue
                
            # Create parser using factory
            parser = ParserFactory.create_parser(file_path, llm_client, logger)
            print(f"\nProcessing {filename} using {parser.__class__.__name__}...")
            
            # Parse the document
            result = parser.parse(file_path)
            
            # Store result in the dictionary
            all_results[filename] = result
            
            # Print summary of extracted content
            print(f"Successfully extracted content from {filename}")
            print(f"Number of chunks/pages: {len(result)}")
            
            # Print short sample of the first chunk
            if result:
                sample_text = result[0].get("text", "")[:100]
                print(f"Sample text: {sample_text}...")
                
        except Exception as e:
            # Log processing errors
            print(f"Error processing {filename}: {str(e)}")
    
    # Save results to JSON file if output path is provided
    if output_path and all_results:
        try:
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
            print(f"\nResults successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    # Example 1: Process a single image with LM Studio
    # print("\n=== Example 1: Process a Single Image with LM Studio ===")
    lmstudio_config = {
        # "base_url": "http://localhost:1234",
        "base_url": os.environ.get("LM_SUDIO"),
        "supports_vision": True  # Force vision support
    }
    # image_result = process_image(
    #     image_path="path/to/your/image.jpg",
    #     client_type="lmstudio",
    #     output_path="lmstudio_image_result.json",
    #     **lmstudio_config
    # )
    # print(f"Extracted text: {image_result[:150]}...")

    # Example 2: Process a document with LM Studio
    print("\n=== Example 2: Process a Document with LM Studio ===")
    # document_result = process_document(
    #     file_path="./Functional Requirements.docx",
    #     client_type="lmstudio",
    #     output_path="lmstudio_document_result.json",
    #     **lmstudio_config
    # )
    
    
    azure_config = {
        "azure_endpoint": os.getenv("AZURE_BASE_ENDPOINT"),
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_version": os.getenv("AZURE_API_VERSION")
    }
    azure_result = process_document(
        file_path="./Functional Requirements.docx",
        client_type="azure",
        output_path="azure_result.json",
        **azure_config
    )
    

    # To use with a different client, just change the client_type and config:
    
    # Example with Azure client (if needed)
    """
    azure_config = {
        "azure_endpoint": os.getenv("AZURE_BASE_ENDPOINT"),
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_version": os.getenv("AZURE_API_VERSION")
    }
    azure_result = process_document(
        file_path="path/to/your/document.pdf",
        client_type="azure",
        output_path="azure_result.json",
        **azure_config
    )
    """
    
    # Example with Ollama client (if needed)
    """
    ollama_config = {
        "host_url": "http://localhost:11434",
        "model_name": "llama3.2"
    }
    ollama_result = process_document(
        file_path="path/to/your/document.pdf",
        client_type="ollama",
        output_path="ollama_result.json",
        **ollama_config
    )
    """