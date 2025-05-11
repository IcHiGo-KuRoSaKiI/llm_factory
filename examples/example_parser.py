import os
import json
from parsers.factory import ParserFactory

# Mock OpenAI helper for example purposes
class MockOpenAIHelper:
    def get_openai_response_image(self, base64_image):
        # In a real scenario, this would call OpenAI's API
        return "This is a mock text extracted from an image using OpenAI's Vision API."

# Mock logger for example purposes
class MockLogger:
    def log_success(self, file_path, schema_name, weaviate_uuid, operation_type):
        print(f"SUCCESS: {operation_type} operation on {file_path} successful.")
        
    def log_failure(self, file_path, error, error_type):
        print(f"FAILURE: {error_type} on {file_path}: {error}")

def process_documents(directory_path, output_json_path=None):
    """
    Process all supported documents in a directory using the parser factory.
    
    Args:
        directory_path: Path to the directory containing documents
        output_json_path: Optional path to save results as JSON
    """
    # Create mock dependencies
    openai_helper = MockOpenAIHelper()
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
            # Create parser using factory
            parser = ParserFactory.create_parser(file_path, openai_helper, logger)
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
                
        except ValueError as e:
            # Skip unsupported file types
            print(f"Skipping {filename}: {str(e)}")
        except Exception as e:
            # Log general processing errors
            print(f"Error processing {filename}: {str(e)}")
    
    # Save results to JSON file if output path is provided
    if output_json_path and all_results:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
            print(f"\nResults successfully saved to {output_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    # Example usage
    # process_documents("path/to/documents", "output_results.json")
    
    # Example of processing a specific file
    try:
        # Initialize dependencies
        openai_helper = MockOpenAIHelper()
        logger = MockLogger()
        
        # Sample file path - adjust as needed
        file_path = "/Users/ichigo/Documents/GitHub/llm_factory/Functional Requirements.docx"
        
        # Create parser using factory
        parser = ParserFactory.create_parser(file_path, openai_helper, logger)
        
        # Parse document
        result = parser.parse(file_path)
        
        # Save results to JSON file
        output_json_path = "/Users/ichigo/Documents/GitHub/llm_factory/parsing_results.json"
        try:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(result, json_file, indent=2, ensure_ascii=False)
            print(f"\nResults successfully saved to {output_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")
        
        # Process extracted content
        for chunk in result:
            page_num = chunk.get("page_number")
            text = chunk.get("text")
            print(f"Page {page_num}: {text[:100]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")