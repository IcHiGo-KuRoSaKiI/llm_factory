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
                 base_url: str = None,
                 api_path: str = "/v1/chat/completions",
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000,
                 supports_vision: bool = True):  # Default to True to assume vision support
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
            self.base_url = base_url or "http://localhost:1234"
            self.base_url = self.base_url.rstrip('/')
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
            logger.warning(f"Warning during LM Studio client initialization: {str(e)}")
            logger.warning("Will attempt to connect during actual requests.")
    
    def _verify_connection(self):
        """Verify connection to LM Studio server"""
        try:
            # Try a simple request to check if the server is available
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code != 200:
                logger.warning(f"LM Studio server connection warning: Status code {response.status_code}")
                logger.warning("The server is available but returned an unexpected status code. Continuing anyway.")
            else:
                logger.info(f"Successfully connected to LM Studio server at {self.base_url}")
                
            # Check if there's information about vision capabilities in the models
            try:
                model_info = response.json()
                if isinstance(model_info, dict) and "data" in model_info:
                    for model in model_info["data"]:
                        if model.get("capabilities") and "vision" in model.get("capabilities", []):
                            logger.info(f"Found vision-capable model: {model.get('id')}")
                            self.supports_vision = True
                            return
            except:
                # If we can't parse model info, just continue with the vision setting from initialization
                pass
            
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to connect to LM Studio server at {self.base_url}")
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
    
# Update the generate_completion method in lmstudio_client.py

    def generate_completion(self,
                        messages: Union[List[Dict[str, str]], str, Dict[str, str]],
                        temperature: float = None,
                        max_tokens: int = None,
                        response_format: Optional[Dict[str, Any]] = None,
                        max_attempts: int = 3,
                        subsection_name: str = "Unknown",
                        conversation_id: str = "default",
                        **kwargs) -> Union[str, Dict]:
        attempts = 0

        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        if isinstance(messages, str):
            messages = self.parse_messages_json(messages)
        elif isinstance(messages, dict):
            messages = [messages]

        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        if (self.conversation_history[conversation_id] and 
            not (len(messages) > 0 and messages[0].get("role") == "system")):
            all_messages = self.conversation_history[conversation_id] + messages
        else:
            all_messages = messages

        while attempts < max_attempts:
            try:
                payload = {
                    "messages": all_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                # Handle response_format
                if response_format:
                    fmt_type = response_format.get("type")
                    if fmt_type == "json_object":
                        payload["response_format"] = {"type": "json_object"}
                    elif fmt_type == "json_schema":
                        schema = response_format.get("json_schema")
                        lm_studio_schema = {
                            "name": "output_schema",
                            "strict": True,
                            "schema": schema
                        }
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": lm_studio_schema
                        }

                for key, value in kwargs.items():
                    if key not in payload and key not in ['subsection_name', 'conversation_id']:
                        payload[key] = value

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code != 200:
                    logger.warning(f"LM Studio API error (attempt {attempts+1}): Status code {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                    raise Exception(f"API returned status code {response.status_code}")

                response_data = response.json()

                if 'choices' in response_data and len(response_data['choices']) > 0:
                    if 'message' in response_data['choices'][0]:
                        content = response_data['choices'][0]['message'].get('content', '')
                    else:
                        content = response_data['choices'][0].get('text', '')
                else:
                    logger.warning("Unexpected response format from LM Studio API")
                    content = str(response_data)

                self.conversation_history[conversation_id] = all_messages.copy()
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": content
                })

                # Process JSON responses when schema was provided
                if response_format and response_format.get("type") == "json_schema":
                    try:
                        # Handle various JSON formats that might come back
                        if isinstance(content, str):
                            # 1. Strip markdown code blocks if present
                            import re
                            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                            if code_block_match:
                                content = code_block_match.group(1)
                            
                            # 2. Remove any comments
                            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                            
                            # 3. Parse the cleaned JSON
                            return json.loads(content)
                        elif isinstance(content, dict):
                            # Already a JSON object
                            return content
                    except json.JSONDecodeError as e:
                        logger.warning(f"Response is not valid JSON despite json_schema format: {e}")
                        logger.warning(f"Raw content: {content[:200]}...")
                        return {"error": "Invalid JSON response", "content": content}
                
                return content

            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)

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
        # Ensure image_data is properly formatted
        if not image_data.startswith("data:"):
            # Convert base64 string to data URI
            image_data = f"data:image/jpeg;base64,{image_data}"
            
        # Use default prompt if none provided
        if not prompt:
            prompt = """
            You are an advanced AI system specializing in extracting and structuring data from images.
            Extract all text content from this image accurately. If the image contains diagrams,
            charts, or tables, describe their structure and content clearly.
            Ignore any watermarks or unrelated background elements.
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
        
        try:
            # Prepare request payload
            payload = {
                "messages": messages,
                "temperature": self.default_temperature,
                "max_tokens": self.default_max_tokens
            }
            
            # Log attempt to use LM Studio for vision
            logger.info(f"Attempting to process image with LM Studio at {self.base_url}")
            
            # Make the API request with a longer timeout for image processing
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            # Check for successful response
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
                    raise Exception(f"Unexpected response format: {response_data}")
            else:
                # Log the error response
                logger.error(f"LM Studio image processing failed with status {response.status_code}: {response.text}")
                raise Exception(f"LM Studio API returned status code {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to process image with LM Studio: {str(e)}")
            
            # Return a clear error message instead of trying OpenAI fallback
            return f"Image processing failed with LM Studio: {str(e)}. Please ensure your LM Studio model supports vision."