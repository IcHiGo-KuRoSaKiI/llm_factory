# clients/openrouter_client.py
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

from llm_factory.clients.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenRouterLLMClient(BaseLLMClient):
    """OpenRouter LLM client implementation using OpenAI SDK"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000,
                 default_top_p: float = 1.0,
                 default_top_k: int = 0,
                 default_frequency_penalty: float = 0.0,
                 default_presence_penalty: float = 0.0,
                 default_repetition_penalty: float = 1.0,
                 default_min_p: float = 0.0,
                 default_top_a: float = 0.0,
                 default_seed: Optional[int] = None):
        """Initialize OpenRouter client using OpenAI SDK"""
        try:
            # Import OpenAI SDK
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with 'pip install openai'")

            # Initialize the client settings
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

            self.model_name = model_name or os.getenv(
                "OPENROUTER_MODEL_NAME", "qwen/qwen3-235b-a22b")

            # Initialize OpenAI client with OpenRouter base URL
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

            # Default parameters
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            self.default_top_p = default_top_p
            self.default_top_k = default_top_k
            self.default_frequency_penalty = default_frequency_penalty
            self.default_presence_penalty = default_presence_penalty
            self.default_repetition_penalty = default_repetition_penalty
            self.default_min_p = default_min_p
            self.default_top_a = default_top_a
            self.default_seed = default_seed

            # Initialize conversation history storage
            self.conversation_history = {}

            # Set up extra headers for OpenRouter
            self.extra_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/IcHiGo-KuRoSaKiI/llm_factory"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "LLM Factory")
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenRouter client: {str(e)}")

    @staticmethod
    def extract_json_from_response(content: str) -> dict:
        """
        Extract and parse JSON from various response formats including markdown code blocks
        """
        import re
        
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
        # Try direct JSON parsing first
        content_stripped = content.strip()
        if content_stripped.startswith('{') and content_stripped.endswith('}'):
            try:
                return json.loads(content_stripped)
            except json.JSONDecodeError:
                pass  # Continue to other parsing methods
        
        # Extract JSON from markdown code blocks
        # Pattern 1: ```json ... ```
        json_pattern = r'```(?:json)?\s*\n?({.*?})\s*\n?```'
        match = re.search(json_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Look for any JSON object in the content
        json_object_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
        matches = re.findall(json_object_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Try to parse each potential JSON object
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Pattern 3: Extract content between first { and last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = content[first_brace:last_brace + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                pass
        
        # If all else fails, try to clean up common formatting issues
        cleaned_content = content.strip()
        # Remove markdown formatting
        cleaned_content = re.sub(r'```(?:json)?\s*\n?', '', cleaned_content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'\n?```\s*$', '', cleaned_content)
        # Remove extra whitespace and newlines
        cleaned_content = re.sub(r'\n\s*', ' ', cleaned_content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not extract valid JSON from response. Content preview: {content[:200]}...") from e

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
                            top_p: Optional[float] = None,
                            top_k: Optional[int] = None,
                            frequency_penalty: Optional[float] = None,
                            presence_penalty: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            min_p: Optional[float] = None,
                            top_a: Optional[float] = None,
                            seed: Optional[int] = None,
                            logit_bias: Optional[Dict[str, float]] = None,
                            response_format: Optional[Dict[str, Any]] = None,
                            max_attempts: int = 3,
                            subsection_name: str = "Unknown",
                            conversation_id: str = "default",
                            **kwargs) -> Union[str, Dict]:
        """Generate a completion using OpenRouter via OpenAI SDK"""
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
                # Build completion parameters
                completion_params = {
                    "model": self.model_name,
                    "messages": all_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_headers": self.extra_headers
                }

                # Add OpenRouter-specific parameters with defaults
                if top_p is not None:
                    completion_params["top_p"] = top_p
                elif self.default_top_p != 1.0:
                    completion_params["top_p"] = self.default_top_p

                if frequency_penalty is not None:
                    completion_params["frequency_penalty"] = frequency_penalty
                elif self.default_frequency_penalty != 0.0:
                    completion_params["frequency_penalty"] = self.default_frequency_penalty

                if presence_penalty is not None:
                    completion_params["presence_penalty"] = presence_penalty
                elif self.default_presence_penalty != 0.0:
                    completion_params["presence_penalty"] = self.default_presence_penalty

                if seed is not None:
                    completion_params["seed"] = seed
                elif self.default_seed is not None:
                    completion_params["seed"] = self.default_seed

                if logit_bias is not None:
                    completion_params["logit_bias"] = logit_bias

                # Handle response format if provided
                if response_format and response_format.get("type") == "json_object":
                    completion_params["response_format"] = {
                        "type": "json_object"}

                # Add any additional parameters compatible with OpenAI SDK
                for key, value in kwargs.items():
                    if key not in completion_params and key not in ['top_k', 'repetition_penalty', 'min_p', 'top_a']:
                        completion_params[key] = value

                # Remove None values
                completion_params = {
                    k: v for k, v in completion_params.items() if v is not None}

                # Debug: Log the request being made (only in debug mode)
                logger.debug(
                    f"OpenRouter API request via OpenAI SDK: {completion_params}")

                # Call the OpenAI API (which will route through OpenRouter)
                response = self.client.chat.completions.create(
                    **completion_params)

                # Extract the content from the response
                content = response.choices[0].message.content

                # Update conversation history
                self.conversation_history[conversation_id] = all_messages.copy(
                )
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": content
                })

                # Return parsed JSON if json response format is requested, otherwise return text
                if response_format and response_format.get("type") in ["json_schema", "json_object"]:
                    try:
                        # Use robust JSON extraction that handles markdown code blocks
                        logger.debug(f"Parsing JSON response with format: {response_format.get('type')}")
                        parsed_result = self.extract_json_from_response(content)
                        logger.debug(f"Successfully parsed JSON with keys: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}")
                        return parsed_result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Could not parse JSON response: {str(e)}")
                        logger.debug(f"Raw content that failed to parse: {content}")
                        return {"error": "Invalid JSON response", "content": content, "parse_error": str(e)}
                else:
                    return content

            except Exception as e:
                logger.warning(
                    f"Error in generation (attempt {attempts+1}): {str(e)}")
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
        Extract text from an image using OpenRouter's Vision capabilities via OpenAI SDK

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

            # Use specified model or default vision model
            vision_model = model or os.getenv(
                "OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")

            # Call the OpenAI API with the image (which will route through OpenRouter)
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=[
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
                                    "url": image_data,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                extra_headers=self.extra_headers
            )

            # Extract and clean the response
            raw_text = response.choices[0].message.content
            cleaned_text = self.clean_extracted_text(raw_text)
            return cleaned_text

        except Exception as e:
            logger.error(f"Error processing image with OpenRouter: {str(e)}")
            return f"[Image processing failed: {str(e)}]"
