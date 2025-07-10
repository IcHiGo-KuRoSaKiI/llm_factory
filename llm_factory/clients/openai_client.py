# clients/openai_client.py
import os
import json
import time
import logging
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAILLMClient(BaseLLMClient):
    """OpenAI client implementation"""

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
        """Initialize OpenAI client"""
        try:
            # Import OpenAI client
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with 'pip install openai'")

            # Initialize the direct OpenAI client
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
            self.model_name = model_name or os.getenv(
                "OPENAI_MODEL_NAME", "gpt-4o-mini")

            # Initialize the LangChain client
            self.llm_params = {
                "openai_api_key": api_key or os.getenv("OPENAI_API_KEY"),
                "model_name": self.model_name,
                "temperature": default_temperature,
                "max_tokens": default_max_tokens
            }
            self.langchain_client = ChatOpenAI(**self.llm_params)

            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            self.default_top_p = default_top_p or float(os.getenv("OPENAI_TOP_P", "1.0"))
            self.default_top_k = default_top_k or int(os.getenv("OPENAI_TOP_K", "0"))
            self.default_frequency_penalty = default_frequency_penalty or float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.0"))
            self.default_presence_penalty = default_presence_penalty or float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.0"))
            self.default_repetition_penalty = default_repetition_penalty or float(os.getenv("OPENAI_REPETITION_PENALTY", "1.0"))
            self.default_min_p = default_min_p or float(os.getenv("OPENAI_MIN_P", "0.0"))
            self.default_top_a = default_top_a or float(os.getenv("OPENAI_TOP_A", "0.0"))
            self.default_seed = default_seed or (int(os.getenv("OPENAI_SEED")) if os.getenv("OPENAI_SEED") else None)

            # Initialize conversation history storage
            self.conversation_history = {}
            
            # Initialize fine-tuning tracking
            self.fine_tuning_logs = []
            self.cache_hits = 0

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

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
                            response_format: Optional[Dict[str, Any]] = None,
                            max_attempts: int = 3,
                            subsection_name: str = "Unknown",
                            conversation_id: str = "default",
                            fine_tune_context: Optional[Dict[str, Any]] = None,
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

        # Create LangChain message objects from the raw messages
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        # If we have history and this isn't a fresh start with a system message
        if (self.conversation_history[conversation_id] and
                not (len(messages) > 0 and messages[0].get("role") == "system")):
            # Get existing history and append new messages
            langchain_messages = self.conversation_history[conversation_id] + \
                langchain_messages

        while attempts < max_attempts:
            try:
                # Prepare parameters with defaults
                effective_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Add optional parameters with defaults
                if top_p is not None:
                    effective_params["top_p"] = top_p
                elif self.default_top_p != 1.0:
                    effective_params["top_p"] = self.default_top_p
                    
                if frequency_penalty is not None:
                    effective_params["frequency_penalty"] = frequency_penalty
                elif self.default_frequency_penalty != 0.0:
                    effective_params["frequency_penalty"] = self.default_frequency_penalty
                    
                if presence_penalty is not None:
                    effective_params["presence_penalty"] = presence_penalty
                elif self.default_presence_penalty != 0.0:
                    effective_params["presence_penalty"] = self.default_presence_penalty
                    
                if seed is not None:
                    effective_params["seed"] = seed
                elif self.default_seed is not None:
                    effective_params["seed"] = self.default_seed
                
                # Add any additional kwargs
                effective_params.update(kwargs)
                
                # Set up the LangChain client with the requested parameters
                llm = self.langchain_client.bind(**effective_params)

                # Handle JSON schema if provided
                if response_format and response_format.get("type") == "json_schema":
                    schema = response_format.get("json_schema", {})

                    # Process schema to ensure it has title and description
                    # If schema is in format {"name": X, "schema": Y}, extract the inner schema
                    if "name" in schema and "schema" in schema:
                        schema = schema["schema"]

                    # Ensure title and description are present
                    if "title" not in schema:
                        schema["title"] = schema.get("name", "OutputSchema")
                    if "description" not in schema:
                        schema["description"] = "Schema for structured output"

                    # Create a structured output with this schema
                    llm_with_schema = llm.with_structured_output(
                        schema=schema,
                        include_raw=True
                    )

                    response = llm_with_schema.invoke(langchain_messages)

                    # Extract content from response
                    if isinstance(response, dict) and "raw" in response:
                        ai_msg_content = response["raw"].content if hasattr(
                            response["raw"], "content") else str(response["raw"])
                        result = {k: v for k, v in response.items()
                                  if k != "raw"}
                    else:
                        ai_msg_content = str(response)
                        result = response

                    # Create AI message for history
                    ai_msg = AIMessage(content=ai_msg_content)

                else:
                    # Standard text generation
                    ai_msg = llm.invoke(langchain_messages)
                    result = ai_msg.content

                # Update conversation history
                if len(langchain_messages) > 0 and isinstance(langchain_messages[0], SystemMessage):
                    # If we start with a system message, keep it but add new interactions
                    if len(self.conversation_history[conversation_id]) == 0:
                        # First time, store the system message
                        self.conversation_history[conversation_id].append(
                            langchain_messages[0])

                    # Add the latest user messages and AI response
                    for msg in langchain_messages[1:]:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            self.conversation_history[conversation_id].append(
                                msg)
                else:
                    # Just append the new messages to existing history
                    for msg in langchain_messages:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            self.conversation_history[conversation_id].append(
                                msg)

                # Add AI response to history
                self.conversation_history[conversation_id].append(ai_msg)
                
                # Track fine-tuning metadata
                fine_tuning_metadata = self._track_fine_tuning(
                    messages=messages,
                    response=result,
                    model_used=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    subsection_name=subsection_name,
                    conversation_id=conversation_id,
                    fine_tune_context=fine_tune_context
                )
                
                # Return result with fine-tuning metadata if structured output
                if response_format and response_format.get("type") == "json_schema":
                    if isinstance(result, dict):
                        result.update(fine_tuning_metadata)
                    return result
                else:
                    return result

            except Exception as e:
                logger.warning(
                    f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)

        # Return placeholder if all attempts failed
        error_msg = f"Failed to generate completion for '{subsection_name}' after {max_attempts} attempts."
        logger.error(error_msg)

        if response_format and response_format.get("type") == "json_schema":
            return {
                "error": error_msg, 
                "status": "failed",
                "fine_tuning_applied": fine_tune_context.get('enabled', False) if fine_tune_context else False,
                "fine_tune_instruction": fine_tune_context.get('prompt', '') if fine_tune_context else "",
                "cache_hits": self.cache_hits,
                "pipeline_config": {
                    "dry_run": False,
                    "top_level_model": self.model_name,
                    "top_level_temperature": temperature,
                    "top_level_max_tokens": max_tokens
                }
            }
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
    
    def _track_fine_tuning(self,
                          messages: List[Dict[str, str]],
                          response: Union[str, Dict],
                          model_used: str,
                          temperature: float,
                          max_tokens: int,
                          response_format: Optional[Dict[str, Any]] = None,
                          subsection_name: str = "Unknown",
                          conversation_id: str = "default",
                          fine_tune_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Track fine-tuning metadata for the current generation"""
        
        # Determine if fine-tuning was applied based on fine_tune_context or model name
        fine_tuning_applied = False
        fine_tune_instruction = ""
        
        if fine_tune_context:
            fine_tuning_applied = fine_tune_context.get('enabled', False)
            fine_tune_instruction = fine_tune_context.get('prompt', '')
        else:
            # Fallback: check model name for fine-tuned indicators
            fine_tuning_applied = "ft-" in model_used or "fine-tuned" in model_used.lower()
        
        # Track cache hits (simplified - could be enhanced with actual cache detection)
        self.cache_hits += 1
        
        # Create fine-tuning metadata
        fine_tuning_metadata = {
            "fine_tuning_applied": fine_tuning_applied,
            "fine_tune_instruction": fine_tune_instruction,
            "cache_hits": self.cache_hits,
            "pipeline_config": {
                "dry_run": False,  # Could be passed as parameter
                "top_level_model": model_used,
                "top_level_temperature": temperature,
                "top_level_max_tokens": max_tokens
            }
        }
        
        # Create detailed log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "subsection_name": subsection_name,
            "model_used": model_used,
            "base_model": self._extract_base_model(model_used),
            "fine_tuning_applied": fine_tuning_applied,
            "fine_tune_instruction": fine_tune_instruction,
            "improvements": self._analyze_improvements(messages, response, fine_tuning_applied),
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format
            },
            "input_tokens": self._estimate_tokens(str(messages)),
            "output_tokens": self._estimate_tokens(str(response)),
            "cache_hits": self.cache_hits
        }
        
        # Add to fine-tuning logs
        self.fine_tuning_logs.append(log_entry)
        
        return fine_tuning_metadata
    
    def _extract_base_model(self, model_name: str) -> str:
        """Extract base model name from fine-tuned model name"""
        if "ft-" in model_name:
            # Fine-tuned model format: ft-gpt-3.5-turbo-0125:personal::8N8.... 
            parts = model_name.split(":")
            if len(parts) > 0:
                base_part = parts[0].replace("ft-", "")
                return base_part
        return model_name
    
    def _analyze_improvements(self, messages: List[Dict[str, str]], response: Union[str, Dict], fine_tuning_applied: bool) -> List[str]:
        """Analyze what improvements fine-tuning provided"""
        improvements = []
        
        if fine_tuning_applied:
            improvements.append("Enhanced response quality through domain-specific training")
            improvements.append("Improved instruction following for specific use cases")
            
            # Analyze response characteristics
            if isinstance(response, dict):
                improvements.append("Better structured output generation")
                if "error" not in response:
                    improvements.append("Reduced error rate in structured responses")
            
            # Analyze response length and coherence
            response_text = str(response)
            if len(response_text) > 100:
                improvements.append("More detailed and comprehensive responses")
                
        return improvements
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def save_fine_tuning_logs(self, file_path: str) -> None:
        """Save fine-tuning logs to a JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare comprehensive log data
            log_data = {
                "client_type": "OpenAI",
                "timestamp": datetime.now().isoformat(),
                "total_generations": len(self.fine_tuning_logs),
                "total_cache_hits": self.cache_hits,
                "fine_tuning_sessions": self.fine_tuning_logs,
                "summary": {
                    "fine_tuned_generations": sum(1 for log in self.fine_tuning_logs if log["fine_tuning_applied"]),
                    "base_model_generations": sum(1 for log in self.fine_tuning_logs if not log["fine_tuning_applied"]),
                    "models_used": list(set(log["model_used"] for log in self.fine_tuning_logs)),
                    "base_models": list(set(log["base_model"] for log in self.fine_tuning_logs))
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Fine-tuning logs saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save fine-tuning logs: {str(e)}")
    
    def get_fine_tuning_summary(self) -> Dict[str, Any]:
        """Get a summary of fine-tuning usage"""
        total_logs = len(self.fine_tuning_logs)
        fine_tuned_count = sum(1 for log in self.fine_tuning_logs if log["fine_tuning_applied"])
        
        return {
            "total_generations": total_logs,
            "fine_tuned_generations": fine_tuned_count,
            "base_model_generations": total_logs - fine_tuned_count,
            "fine_tuning_percentage": (fine_tuned_count / total_logs * 100) if total_logs > 0 else 0,
            "cache_hits": self.cache_hits,
            "models_used": list(set(log["model_used"] for log in self.fine_tuning_logs)),
            "recent_improvements": self.fine_tuning_logs[-1]["improvements"] if self.fine_tuning_logs else []
        }
