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
                 default_max_tokens: int = 8000):
        """
        Initialize LM Studio client
        
        Args:
            base_url: Base URL of the LM Studio server (e.g., "http://localhost:1234")
            api_path: API endpoint path (default: "/v1/chat/completions")
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
        """
        try:
            # Store configuration
            self.base_url = base_url.rstrip('/')
            self.api_path = api_path
            self.api_url = f"{self.base_url}{self.api_path}"
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            
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