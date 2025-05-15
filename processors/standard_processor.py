# processors/standard_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BasePromptProcessor

logger = logging.getLogger(__name__)


class StandardPromptProcessor(BasePromptProcessor):
    """
    Processor for standard prompts with JSON extraction and optional tonality matching.
    Handles traditional extraction + tonality workflow.
    """

    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process a standard prompt with optional JSON schema and tonality matching.

        Args:
            client: The LLM client to use for generation
            prompt_config: The prompt configuration dictionary containing:
                - name: Name/identifier for this extraction
                - prompt: The extraction prompt
                - schema: JSON schema (Optional)
                - tonality_messages: Base tonality matching messages (Optional)
                - context_data: Context data related to this extraction
            **kwargs: Additional parameters for the LLM client

        Returns:
            Dictionary containing results
        """
        try:
            # Check if we're dealing with a single extraction or multiple extractions
            if isinstance(prompt_config, list):
                # Process multiple extractions
                return self.process_traditional_extractions(
                    client=client,
                    extraction_configs=prompt_config,
                    **kwargs
                )
            else:
                # Process a single extraction
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
                    raise ValueError(
                        f"Missing required 'prompt' in prompt configuration for '{name}'")

                # Generate a unique conversation ID
                conversation_id = f"{name}_{id(context_data or prompt)}"

                # Create a copy of kwargs without temperature and max_tokens
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in [
                    'temperature', 'max_tokens']}

                # Process the extraction prompt
                json_result, tonality_msgs, tonality_result = self._process_extraction_with_tonality(
                    client=client,
                    extraction_prompt=prompt,
                    context_data=context_data,
                    json_schema=schema,
                    tonality_messages=tonality_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    subsection_name=name,
                    conversation_id=conversation_id,
                    **filtered_kwargs
                )

                # Return the results
                result = {name: {}}

                # If schema was provided, use json_result key, otherwise use result key
                if schema:
                    result[name]['json_result'] = json_result
                else:
                    result[name]['result'] = json_result

                # Only add tonality information if tonality messages were provided
                if tonality_messages and len(tonality_messages) > 0:
                    result[name]['tonality_messages'] = tonality_msgs
                    result[name]['tonality_result'] = tonality_result

                return result

        except Exception as e:
            logger.error(f"Error processing standard prompt: {str(e)}")
            return {
                'error': {
                    'success': False,
                    'message': f"Error processing standard prompt: {str(e)}"
                }
            }

    def process_traditional_extractions(
        self,
        client,
        extraction_configs: List[Dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 8000,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple extractions with the traditional two-step approach

        Args:
            client: The LLM client to use for generation
            extraction_configs: List of dictionaries containing:
                - name: Name/identifier for this extraction
                - prompt: The extraction prompt
                - schema: JSON schema (Optional)
                - tonality_messages: Base tonality matching messages (Optional)
                - context_data: Context data related to this extraction
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            **kwargs: Additional parameters for the LLM client

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing results for each extraction
        """
        results = {}

        try:
            for config in extraction_configs:
                name = config.get('name', 'unnamed_extraction')
                context_data = config.get('context_data', None)
                schema = config.get('schema', None)

                # Get subsection name for better error reporting
                subsection_name = name

                if context_data is None:
                    logger.warning(
                        f"⚠️ Context data is missing for extraction: {subsection_name}")
                    results[name] = {
                        'result': {"status": "failed", "error": "Missing context data"}
                    }
                    continue

                # Generate a unique conversation ID for this extraction
                conversation_id = f"{name}_{id(context_data)}"

                # Create a copy of kwargs without temperature and max_tokens
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in [
                    'temperature', 'max_tokens']}

                # Process extraction and tonality with conversation history
                json_result, tonality_messages, tonality_result = self._process_extraction_with_tonality(
                    client=client,
                    extraction_prompt=config['prompt'],
                    json_schema=config.get('schema', None),
                    tonality_messages=config.get('tonality_messages', None),
                    context_data=context_data,
                    subsection_name=subsection_name,
                    conversation_id=conversation_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **filtered_kwargs
                )

                # Store results
                results[name] = {}

                # If schema was provided, use json_result key, otherwise use result key
                if schema:
                    results[name]['json_result'] = json_result
                else:
                    results[name]['result'] = json_result

                # Only add tonality information if tonality messages were provided
                if config.get('tonality_messages') and len(config.get('tonality_messages', [])) > 0:
                    results[name]['tonality_messages'] = tonality_messages
                    results[name]['tonality_result'] = tonality_result

                logger.info(
                    f"✅ Completed extraction and tonality for '{subsection_name}'")

            return results

        except Exception as e:
            logger.error(f"❌ Error in extraction processing: {str(e)}")
            return results

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
    ) -> Tuple[Any, List[Dict[str, str]], str]:
        """
        Process extraction with JSON schema followed by tonality matching
        using conversation history between steps

        Args:
            client: The LLM client to use for generation
            extraction_prompt: The initial extraction prompt
            context_data: The context data to process
            json_schema: JSON schema for structured extraction
            tonality_messages: Base tonality matching messages
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            subsection_name: Name of the subsection
            conversation_id: Optional ID for this conversation thread
            **kwargs: Additional parameters for the LLM client

        Returns:
            Tuple[Any, List[Dict[str, str]], str]: Extracted JSON, tonality messages, tonality result
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
                extraction_messages = [
                    extraction_system_msg, extraction_user_msg]
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
                    response_format={"type": "json_schema",
                                     "json_schema": json_schema},
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
                return extracted_json, [], "No tonality matching available."

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
                "content": f"Convert the following information to a standardized format: '''{json_str} ''' "
            })

            # Process tonality using a different conversation ID to avoid mixing contexts
            tonality_result = client.generate_completion(
                messages=processed_tonality_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                subsection_name=f"{subsection_name} (tonality matching)",
                conversation_id=f"{conversation_id}_tonality",
                **kwargs
            )

            return extracted_json, processed_tonality_messages, tonality_result

        except Exception as e:
            logger.error(
                f"Error in extraction and tonality for '{subsection_name}': {str(e)}")

            # Return placeholders instead of raising an exception
            placeholder_json = {
                "status": "failed", "error": f"Failed processing for '{subsection_name}'"}
            placeholder_tonality = f"[Content generation failed for '{subsection_name}']"

            return placeholder_json, [], placeholder_tonality
