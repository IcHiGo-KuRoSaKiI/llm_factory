# clients/groq_client.py
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class GroqLLMClient(BaseLLMClient):
    """Groq LLM client implementation"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000):
        """Initialize Groq client"""
        try:
            # Import Groq client (we import here to make it optional)
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("Groq package not installed. Install with 'pip install groq'")
            
            # Initialize the client
            self.client = Groq(
                api_key=api_key or os.getenv("GROQ_API_KEY")
            )
            self.model_name = model_name or os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            
            # Initialize conversation history storage
            self.conversation_history = {}
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {str(e)}")
    
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
                           temperature: float = 0,
                           max_tokens: int = 8000,
                           response_format: Optional[Dict[str, Any]] = None,
                           max_attempts: int = 3,
                           subsection_name: str = "Unknown",
                           conversation_id: str = "default",
                           **kwargs) -> Union[str, Dict]:
        """Generate a completion using Groq"""
        attempts = 0

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
                completion_params = {
                    "model": self.model_name,
                    "messages": all_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Handle response format if provided
                if response_format and response_format.get("type") == "json_object":
                    completion_params["response_format"] = {"type": "json_object"}
                
                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in completion_params:
                        completion_params[key] = value
                
                # Call the Groq API
                response = self.client.chat.completions.create(**completion_params)
                
                # Extract the content from the response
                content = response.choices[0].message.content
                
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
        Extract text from an image using OpenAI's Vision capabilities
        
        Note: Groq doesn't support vision features natively, so we use OpenAI as a fallback
        
        Args:
            image_data: Base64-encoded image data or data URI
            prompt: Optional custom prompt to use for image analysis
            model: Optional model name to use for image analysis
            
        Returns:
            Extracted text from the image
        """
        try:
            # Import OpenAI client for image processing
            from .openai_client import OpenAILLMClient
            
            logger.info("Using OpenAI client for image processing (Groq doesn't support vision)")
            openai_client = OpenAILLMClient()
            
            # Delegate to OpenAI client
            return openai_client.get_openai_response_image(image_data, prompt, model)
            
        except Exception as e:
            logger.error(f"Error delegating image processing to OpenAI: {str(e)}")
            return f"[Image processing failed: {str(e)}]"