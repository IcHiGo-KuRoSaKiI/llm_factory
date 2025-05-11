# clients/lmstudio_client.py
import json
import time
import logging
import requests
from typing import Any, Dict, List, Optional, Union

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class LMStudioClient(BaseLLMClient):
    """Client for connecting to local LM Studio models"""
    
    def __init__(self,
                 base_url: str,
                 api_path: str = "/v1/chat/completions",
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000,
                 supports_vision: bool = False):
        """
        Initialize LM Studio client
        
        Args:
            base_url: Base URL of the LM Studio server (e.g., "http://localhost:1234")
            api_path: API endpoint path (default: "/v1/chat/completions")
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
            supports_vision: Flag indicating if the loaded model supports vision
        """
        try:
            # Store configuration
            self.base_url = base_url.rstrip('/')
            self.api_path = api_path
            self.api_url = f"{self.base_url}{self.api_path}"
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            self.supports_vision = supports_vision
            
            # Initialize conversation history storage
            self.conversation_history = {}
            
            # Verify connection
            self._verify_connection()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LM Studio client: {str(e)}")
    
    def _verify_connection(self):
        """Verify connection to LM Studio server"""
        try:
            # Try a simple request to check if the server is available
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code != 200:
                logger.warning(f"LM Studio server connection warning: Status code {response.status_code}")
                logger.warning("The server is available but returned an unexpected status code. Continuing anyway.")
            
            # Check if there's information about vision capabilities in the models
            model_info = response.json()
            if isinstance(model_info, dict) and "data" in model_info:
                for model in model_info["data"]:
                    if model.get("capabilities") and "vision" in model.get("capabilities", []):
                        logger.info(f"Found vision-capable model: {model.get('id')}")
                        self.supports_vision = True
                        return
            
        except Exception as e:
            logger.warning(f"Failed to verify LM Studio server connection: {str(e)}")
            logger.warning("Will attempt to connect during actual requests.")
    
    @staticmethod
    def parse_messages_json(messages_json: str) -> List[Dict[str, str]]:
        """Parse a JSON string containing messages into a list of message dictionaries"""
        try:
            if messages_json.strip().startswith('{'):
                messages_json = f"[{messages_json}]"

            messages = json.loads(messages_json)

            if isinstance(messages, dict):
                messages = [messages]

            for message in messages:
                if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                    raise ValueError(
                        "Invalid message format. Each message must have 'role' and 'content' fields.")

            return messages

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing messages: {str(e)}")
    
    def generate_completion(self,
                           messages: Union[List[Dict[str, str]], str, Dict[str, str]],
                           temperature: float = None,
                           max_tokens: int = None,
                           response_format: Optional[Dict[str, Any]] = None,
                           max_attempts: int = 3,
                           subsection_name: str = "Unknown",
                           conversation_id: str = "default",
                           **kwargs) -> Union[str, Dict]:
        """
        Generate a completion using LM Studio local model
        
        Args:
            messages: Messages to send (list of dicts, single dict, or JSON string)
            temperature: Temperature for generation (defaults to self.default_temperature)
            max_tokens: Maximum tokens for response (defaults to self.default_max_tokens)
            response_format: Response format specification
            max_attempts: Number of retries on failure
            subsection_name: Name of the subsection being processed (for logging)
            conversation_id: ID to track conversation thread
            **kwargs: Additional parameters for the API
        
        Returns:
            Union[str, Dict]: Generated completion or structured response
        """
        attempts = 0
        
        # Use default values if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # Handle different message formats
        if isinstance(messages, str):
            messages = self.parse_messages_json(messages)
        elif isinstance(messages, dict):
            messages = [messages]

        # Initialize history for this conversation if not exists
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
            
        # If we have history and this isn't a fresh start with a system message
        if (self.conversation_history[conversation_id] and 
            not (len(messages) > 0 and messages[0].get("role") == "system")):
            # Get existing history and append new messages
            all_messages = self.conversation_history[conversation_id] + messages
        else:
            all_messages = messages
        
        while attempts < max_attempts:
            try:
                # Prepare request payload
                payload = {
                    "messages": all_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Handle response format if provided
                if response_format and response_format.get("type") == "json_object":
                    payload["response_format"] = {"type": "json_object"}
                
                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in payload and key not in ['subsection_name', 'conversation_id']:
                        payload[key] = value
                
                # Make the API request
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Check for successful response
                if response.status_code != 200:
                    logger.warning(f"LM Studio API error (attempt {attempts+1}): Status code {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                    raise Exception(f"API returned status code {response.status_code}")
                
                # Parse the response
                response_data = response.json()
                
                # Extract the content from the response
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    if 'message' in response_data['choices'][0]:
                        content = response_data['choices'][0]['message'].get('content', '')
                    else:
                        content = response_data['choices'][0].get('text', '')
                else:
                    logger.warning("Unexpected response format from LM Studio API")
                    content = str(response_data)
                
                # Update conversation history
                self.conversation_history[conversation_id] = all_messages.copy()
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": content
                })
                
                # Return parsed JSON if json_object format is used, otherwise return text
                if response_format and response_format.get("type") == "json_schema":
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning("Response is not valid JSON despite json_object format")
                        return {"error": "Invalid JSON response", "content": content}
                else:
                    return content
                
            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)
        
        # Return placeholder if all attempts failed
        error_msg = f"Failed to generate completion for '{subsection_name}' after {max_attempts} attempts."
        logger.error(error_msg)
        
        if response_format and response_format.get("type") == "json_schema":
            return {"error": error_msg, "status": "failed"}
        else:
            return f"[Content generation failed: {error_msg}]"
    
    def get_openai_response_image(self, 
                                 image_data: str, 
                                 prompt: Optional[str] = None,
                                 model: Optional[str] = None) -> str:
        """
        Extract text from an image using LM Studio's vision capabilities
        
        Args:
            image_data: Base64-encoded image data or data URI
            prompt: Optional custom prompt to use for image analysis
            model: Optional model name to use for image analysis
            
        Returns:
            Extracted text from the image
        """
        try:
            # First try to use the local LM Studio server with vision capabilities
            if self.supports_vision:
                # Ensure image_data is properly formatted
                if not image_data.startswith("data:"):
                    # Convert base64 string to data URI
                    image_data = f"data:image/jpeg;base64,{image_data}"
                    
                # Use default prompt if none provided
                if not prompt:
                    prompt = """
                    You are an advanced AI system specializing in extracting and structuring data from images while adhering to strict operational criteria. Follow these directives meticulously:
                    1. **Basic Image Assessment:**
                    - If the image contains only one logo, return an empty response.
                    - If the image does not have business relevance or lacks content related to proposals, pitches, case studies, or similar contexts, return an empty response.
                    2. **Data Extraction Guidelines:**
                    - **Text Content:** Extract the textual content exactly as it appears. Preserve line breaks only if they represent distinct content. Ignore formatting, watermarks, and repetitive elements unless they add meaningful context.
                    - **Company Logos:** If identifiable, extract only the name of the company. Avoid extracting logo images or designs.
                    - **Diagrams:** Be very elaborative when describing diagrams. Properly capture their essence by providing a detailed summary of the content, purpose, and context of the diagram. Highlight key elements, relationships, and insights represented in the diagram.
                    - **Tabular Data:** Convert tabular data into a structured and cleanly formatted output, maintaining clarity.
                    3. **Output Requirements:**
                    - Ensure all extracted data is meaningful, structured, and concise.
                    - Avoid redundancy or inclusion of irrelevant details.
                    4. **Ethical and Privacy Considerations:**
                    - Prioritize user data confidentiality.
                    - Ensure the extracted information aligns with ethical guidelines.
                    Your primary objective is to accurately extract, structure, and interpret relevant data while maintaining a high standard of contextual and ethical awareness.
                    """
                
                # Create request with image in the format matching OpenAI's vision API
                messages = [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            }
                        ]
                    }
                ]
                
                # Try to send a vision request to LM Studio
                logger.info("Attempting to use local LM Studio model for vision processing")
                
                # Prepare request payload
                payload = {
                    "messages": messages,
                    "temperature": self.default_temperature,
                    "max_tokens": self.default_max_tokens
                }
                
                # Make the API request
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Increased timeout for image processing
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Extract the content from the response
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        if 'message' in response_data['choices'][0]:
                            content = response_data['choices'][0]['message'].get('content', '')
                        else:
                            content = response_data['choices'][0].get('text', '')
                            
                        # Clean extracted text
                        return self.clean_extracted_text(content)
                else:
                    # If status code is not 200, the local model might not support vision
                    logger.warning(f"LM Studio API error for vision: Status code {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                    self.supports_vision = False  # Update flag as vision isn't supported
                    raise Exception("Local model does not support vision capabilities")
                        
            # If local model doesn't support vision, fall back to OpenAI
            if not self.supports_vision:
                # Import OpenAI client for image processing
                from .openai_client import OpenAILLMClient
                
                logger.info("Using OpenAI client for image processing (LM Studio model doesn't support vision)")
                openai_client = OpenAILLMClient()
                
                # Delegate to OpenAI client
                return openai_client.get_openai_response_image(image_data, prompt, model)
                
        except Exception as e:
            logger.error(f"Error in vision processing: {str(e)}")
            
            try:
                # Fallback to OpenAI
                from .openai_client import OpenAILLMClient
                
                logger.info("Falling back to OpenAI client after LM Studio vision processing failed")
                openai_client = OpenAILLMClient()
                
                # Delegate to OpenAI client
                return openai_client.get_openai_response_image(image_data, prompt, model)
                
            except Exception as fallback_error:
                logger.error(f"Fallback to OpenAI failed: {str(fallback_error)}")
                return f"[Image processing failed: {str(e)}]"