# clients/openai_client.py
import os
import json
import time
import logging
import base64
from typing import Any, Dict, List, Optional, Union

from jsonschema import validate, ValidationError

from llm_factory.utils import components_relationships_schema

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class OpenAILLMClient(BaseLLMClient):
    """OpenAI client implementation"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000):
        """Initialize OpenAI client"""
        try:
            # Import OpenAI client
            try:
                import openai
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
            
            # Initialize the client
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
            self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            
            # Initialize conversation history storage
            self.conversation_history = {}
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

    @staticmethod
    def extract_json_from_response(content: str) -> dict:
        """Extract and parse JSON from API responses."""
        import re

        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")

        content_stripped = content.strip()
        if content_stripped.startswith('{') and content_stripped.endswith('}'):
            try:
                return json.loads(content_stripped)
            except json.JSONDecodeError:
                pass

        json_pattern = r'```(?:json)?\s*\n?({.*?})\s*\n?```'
        match = re.search(json_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        json_object_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
        matches = re.findall(json_object_pattern, content, re.DOTALL)
        for candidate in matches:
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue

        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = content[first_brace:last_brace + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                pass

        cleaned_content = re.sub(r'```(?:json)?\s*\n?', '', content.strip(), flags=re.IGNORECASE)
        cleaned_content = re.sub(r'\n?```\s*$', '', cleaned_content)
        cleaned_content = re.sub(r'\n\s*', ' ', cleaned_content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        return json.loads(cleaned_content)
    
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
        """Generate a completion using OpenAI"""
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
                if response_format:
                    if response_format.get("type") == "json_object":
                        completion_params["response_format"] = {"type": "json_object"}
                    elif response_format.get("type") == "json_schema":
                        schema = (response_format.get("schema") or
                                  response_format.get("json_schema") or
                                  components_relationships_schema)
                        completion_params["response_format"] = {
                            "type": "json_schema",
                            "schema": schema
                        }
                
                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in completion_params:
                        completion_params[key] = value
                
                # Call the OpenAI API
                response = self.client.chat.completions.create(**completion_params)
                
                # Extract the content from the response
                content = response.choices[0].message.content
                
                # Update conversation history
                self.conversation_history[conversation_id] = all_messages.copy()
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": content
                })
                
                # Process JSON output when requested
                if response_format and response_format.get("type") in ["json_schema", "json_object"]:
                    try:
                        parsed_result = self.extract_json_from_response(content)
                        if response_format.get("type") == "json_schema":
                            schema = (response_format.get("schema") or
                                      response_format.get("json_schema") or
                                      components_relationships_schema)
                            try:
                                validate(instance=parsed_result, schema=schema)
                            except ValidationError as ve:
                                logger.warning(f"Schema validation failed: {ve.message}")
                        return parsed_result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Could not parse JSON response: {str(e)}")
                        logger.debug(f"Raw content that failed to parse: {content}")
                        return {"error": "Invalid JSON response", "content": content, "parse_error": str(e)}
                else:
                    return content
                
            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)
        
        # Return placeholder if all attempts failed
        error_msg = f"Failed to generate completion for '{subsection_name}' after {max_attempts} attempts."
        logger.error(error_msg)
        
        if response_format and response_format.get("type") in ["json_schema", "json_object"]:
            return {"error": error_msg, "status": "failed"}
        else:
            return f"[Content generation failed: {error_msg}]"
    
    def get_openai_response_image(self, 
                                 image_data: str, 
                                 prompt: Optional[str] = None,
                                 model: Optional[str] = None) -> str:
        """
        Extract text from an image using OpenAI's Vision capabilities
        
        Args:
            image_data: Base64-encoded image data or data URI
            prompt: Optional custom prompt to use for image analysis
            model: Optional model name to use for image analysis
            
        Returns:
            Extracted text from the image
        """
        try:
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
            
            # Use specified model or default
            vision_model = model or "gpt-4o-mini"
            
            # Call the OpenAI API with the image
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                                "detail": "high"
                            }
                        }]
                    }
                ],
                model=vision_model
            )
            
            # Extract and clean the response
            raw_text = chat_completion.choices[0].message.content
            cleaned_text = self.clean_extracted_text(raw_text)
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing image with OpenAI: {str(e)}")
            return f"[Image processing failed: {str(e)}]"