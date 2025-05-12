# clients/lmstudio_client.py
import json
import time
import logging
import base64
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class LMStudioClient(BaseLLMClient):
    """Client for connecting to local LM Studio models using the OpenAI SDK"""
    
    def __init__(self,
                 base_url: str = None,
                 api_path: str = "/v1",
                 model_name: str = None,
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000,
                 supports_vision: bool = True,
                 **kwargs):
        """
        Initialize LM Studio client using the OpenAI SDK with a custom base URL
        
        Args:
            base_url: Base URL of the LM Studio server (e.g., "http://localhost:1234")
            api_path: API endpoint path (default: "/v1")
            model_name: Name of the model loaded in LM Studio
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
            supports_vision: Flag indicating if the loaded model supports vision
            **kwargs: Additional parameters for the OpenAI client
        """
        try:
            # Store configuration
            self.base_url = base_url or "http://localhost:1234"
            self.base_url = self.base_url.rstrip('/')
            self.api_path = api_path
            self.full_base_url = f"{self.base_url}{self.api_path}"
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            self.supports_vision = supports_vision
            self.model_name = model_name or "local-model"
            
            # Initialize the OpenAI client with custom base URL
            self.client = OpenAI(
                base_url=self.full_base_url,
                **kwargs
            )
            
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
            models = self.client.models.list()
            logger.info(f"Successfully connected to LM Studio server at {self.base_url}")
            
            # Check if there's information about the active model
            try:
                model_list = self.client.models.list()
                if hasattr(model_list, 'data') and len(model_list.data) > 0:
                    # Use the first model as default if not specified
                    if not self.model_name or self.model_name == "local-model":
                        self.model_name = model_list.data[0].id
                        logger.info(f"Using default model: {self.model_name}")
                    
                    # Check for vision capabilities
                    for model in model_list.data:
                        if hasattr(model, 'capabilities') and 'vision' in getattr(model, 'capabilities', []):
                            logger.info(f"Found vision-capable model: {model.id}")
                            self.supports_vision = True
                            if model.id != self.model_name:
                                logger.info(f"Consider using {model.id} for vision tasks")
            except Exception as e:
                logger.warning(f"Could not retrieve model information: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Failed to connect to LM Studio server at {self.base_url}: {str(e)}")
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
        Generate a completion using LM Studio
        
        Args:
            messages: Messages in various formats
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Response format specification
            max_attempts: Number of retries on failure
            subsection_name: Name of the subsection being processed
            conversation_id: ID to track this conversation thread
            **kwargs: Additional parameters for the API
            
        Returns:
            Union[str, Dict]: Generated completion
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
                # Prepare parameters for the API call
                completion_params = {
                    "model": self.model_name,
                    "messages": all_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Handle response format
                if response_format:
                    if response_format.get("type") == "json_object":
                        completion_params["response_format"] = {"type": "json_object"}
                    elif response_format.get("type") == "json_schema":
                        # Get the schema
                        schema = response_format.get("json_schema", {})
                        
                        # Format the schema according to LM Studio's expected format
                        # LM Studio expects a specific nested format with 'name' and 'schema' fields
                        schema_name = schema.get("title", "output_schema") if isinstance(schema, dict) else "output_schema"
                        
                        # Format schema according to LM Studio's expected structure
                        lm_studio_schema = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": schema
                            }
                        }
                        
                        # Apply the formatted schema
                        completion_params["response_format"] = lm_studio_schema
                        
                        # Also add a hint in the last message to enforce JSON output
                        last_message_index = len(all_messages) - 1
                        if last_message_index >= 0 and "content" in all_messages[last_message_index]:
                            content = all_messages[last_message_index]["content"]
                            if not content.endswith("Respond with a JSON object."):
                                all_messages[last_message_index]["content"] = (
                                    f"{content}\n\nRespond with a structured JSON object that strictly follows "
                                    f"the specified schema. Do not include explanations or markdown formatting."
                                )
                
                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in completion_params and key not in ['subsection_name', 'conversation_id']:
                        completion_params[key] = value
                
                # Make the API call
                response = self.client.chat.completions.create(**completion_params)
                
                # Extract content from the response
                content = response.choices[0].message.content
                
                # Update conversation history
                self.conversation_history[conversation_id] = all_messages.copy()
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": content
                })
                
                # Process JSON responses
                if response_format and (response_format.get("type") == "json_schema" or response_format.get("type") == "json_object"):
                    try:
                        # First check if the content is already valid JSON
                        try:
                            parsed_json = json.loads(content)
                            return parsed_json
                        except json.JSONDecodeError:
                            # If not, try to extract JSON from markdown code blocks
                            import re
                            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                            if code_block_match:
                                json_content = code_block_match.group(1).strip()
                                return json.loads(json_content)
                            
                            # If no code block, try to find JSON-like content (anything between curly braces)
                            json_pattern = re.search(r'(\{[\s\S]*\})', content)
                            if json_pattern:
                                json_content = json_pattern.group(1).strip()
                                return json.loads(json_content)
                            
                            # If all extraction attempts fail, return an error
                            logger.warning(f"Could not extract valid JSON from response: {content[:200]}...")
                            return {"error": "Invalid JSON response", "content": content}
                    except Exception as e:
                        logger.warning(f"Error processing JSON response: {str(e)}")
                        logger.warning(f"Raw content: {content[:200]}...")
                        return {"error": f"JSON processing error: {str(e)}", "content": content}
                    
                return content
                
            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)
        
        # Return placeholder if all attempts failed
        error_msg = f"Failed to generate completion for '{subsection_name}' after {max_attempts} attempts."
        logger.error(error_msg)
        
        if response_format and (response_format.get("type") == "json_schema" or response_format.get("type") == "json_object"):
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
        if not self.supports_vision:
            logger.warning("LM Studio model may not support vision capabilities")
            
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
        
        # Create messages with image content
        try:
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
            
            # Use a longer timeout for image processing
            logger.info("Attempting to process image with LM Studio")
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=model or self.model_name,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
                timeout=60  # Longer timeout for image processing
            )
            
            # Extract the content
            content = response.choices[0].message.content
            
            # Clean the extracted text
            return self.clean_extracted_text(content)
                
        except Exception as e:
            logger.error(f"Failed to process image with LM Studio: {str(e)}")
            
            # Return a clear error message
            return f"Image processing failed with LM Studio: {str(e)}. Please ensure your LM Studio model supports vision."