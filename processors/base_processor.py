# processors/base_processor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BasePromptProcessor(ABC):
    """Base class for prompt processors"""
    
    @abstractmethod
    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a prompt configuration using the provided LLM client"""
        pass

# processors/standard_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Union

from .base_processor import BasePromptProcessor

logger = logging.getLogger(__name__)

class StandardPromptProcessor(BasePromptProcessor):
    """Processor for standard prompts with extraction and tonality matching"""
    
    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a standard prompt with optional JSON schema and tonality matching"""
        try:
            # Extract params from config
            name = prompt_config.get('name', 'unnamed_extraction')
            prompt = prompt_config.get('prompt', '')
            schema = prompt_config.get('schema')
            tonality_messages = prompt_config.get('tonality_messages')
            context_data = prompt_config.get('context_data')
            
            # Get runtime params with defaults from kwargs
            temperature = kwargs.get('temperature', 0)
            max_tokens = kwargs.get('max_tokens', 8000)
            
            # Validate required inputs
            if not prompt:
                raise ValueError(f"Missing required 'prompt' in prompt configuration for '{name}'")
            
            # Generate a unique conversation ID
            conversation_id = f"{name}_{id(context_data or prompt)}"
            
            # Process the extraction prompt
            results = self._process_extraction_with_tonality(
                client=client,
                extraction_prompt=prompt,
                context_data=context_data,
                json_schema=schema,
                tonality_messages=tonality_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                subsection_name=name,
                conversation_id=conversation_id,
                **kwargs
            )
            
            return {
                name: results
            }
            
        except Exception as e:
            logger.error(f"Error processing standard prompt: {str(e)}")
            return {
                'error': {
                    'success': False,
                    'message': f"Error processing standard prompt: {str(e)}"
                }
            }
    
    def _process_extraction_with_tonality(
        self,
        client,
        extraction_prompt: str,
        context_data: Any,
        json_schema: Dict[str, Any] = None,
        tonality_messages: List[Dict[str, str]] = None,
        temperature: float = 0,
        max_tokens: int = 8000,
        subsection_name: str = "Unknown",
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process extraction with JSON schema followed by tonality matching
        """
        try:
            # Generate a conversation ID if not provided
            if conversation_id is None:
                conversation_id = f"{subsection_name}_{id(context_data)}"
            
            # Step 1: JSON Extraction
            logger.info(f"Running JSON extraction for '{subsection_name}'")
            
            # Format context data as string if needed
            if context_data is not None:
                if not isinstance(context_data, str):
                    context_str = json.dumps(context_data)
                else:
                    context_str = context_data
                
                # Create system message for extraction
                extraction_system_msg = {
                    "role": "system",
                    "content": extraction_prompt
                }
                
                # Create user message with context data
                extraction_user_msg = {
                    "role": "user",
                    "content": f"Context Data ``` {context_str} ```"
                }
                
                # Messages for extraction
                extraction_messages = [extraction_system_msg, extraction_user_msg]
            else:
                # If no context data, use the prompt directly
                extraction_messages = [
                    {
                        "role": "system",
                        "content": extraction_prompt
                    }
                ]
            
            # Run extraction with schema
            if json_schema:
                extracted_json = client.generate_completion(
                    messages=extraction_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_schema", "json_schema": json_schema},
                    subsection_name=f"{subsection_name} (JSON extraction)",
                    conversation_id=conversation_id,
                    **kwargs
                )
            else:
                extracted_json = client.generate_completion(
                    messages=extraction_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    subsection_name=f"{subsection_name} (extraction)",
                    conversation_id=conversation_id,
                    **kwargs
                )
            
            # Skip tonality matching if not provided
            if not tonality_messages:
                return {
                    'json_result': extracted_json,
                    'tonality_messages': [],
                    'tonality_result': "No tonality matching available."
                }
            
            # Step 2: Tonality Matching
            logger.info(f"Running tonality matching for '{subsection_name}'")
            
            # Convert extracted JSON to string if it's a dictionary
            if isinstance(extracted_json, dict):
                json_str = json.dumps(extracted_json)
            else:
                json_str = str(extracted_json)
            
            # Prepare tonality messages
            processed_tonality_messages = []
            for msg in tonality_messages:
                processed_tonality_messages.append({
                    "role": msg.get("role", "system"),
                    "content": msg.get("content", "")
                })
            
            # Add the user message with extracted JSON
            processed_tonality_messages.append({
                "role": "user",
                "content": f"Convert the following drug information to a standardized list : '''{json_str} ''' "
            })
            
            # Process tonality using the same conversation ID to maintain history
            tonality_result = client.generate_completion(
                messages=processed_tonality_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                subsection_name=f"{subsection_name} (tonality matching)",
                conversation_id=f"{conversation_id}_tonality",
                **kwargs
            )
            
            return {
                'json_result': extracted_json,
                'tonality_messages': processed_tonality_messages,
                'tonality_result': tonality_result
            }
            
        except Exception as e:
            logger.error(f"Error in extraction and tonality for '{subsection_name}': {str(e)}")
            
            # Return placeholders instead of raising an exception
            placeholder_json = {"status": "failed", "error": f"Failed processing for '{subsection_name}'"}
            placeholder_tonality = f"[Content generation failed for '{subsection_name}']"
            
            return {
                'json_result': placeholder_json,
                'tonality_messages': [],
                'tonality_result': placeholder_tonality
            }
