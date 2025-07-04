# processors/standard_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BasePromptProcessor
try:
    from ..utils.dry_run_logger import create_dry_run_logger
except ImportError:
    # Fallback for direct import
    try:
        from llm_factory.utils.dry_run_logger import create_dry_run_logger
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))
        from dry_run_logger import create_dry_run_logger

logger = logging.getLogger(__name__)


class StandardPromptProcessor(BasePromptProcessor):
    """
    Processor for standard prompts with JSON extraction and optional tonality matching.
    Handles traditional extraction + tonality workflow.
    """
    
    def __init__(self):
        """Initialize the Standard processor with dry-run logging capabilities"""
        self.dry_run_logger = None

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
                - model: Model name (Optional, top-level)
                - temperature: Temperature parameter (Optional, top-level)
                - max_tokens: Max tokens parameter (Optional, top-level)
                - dry_run: Enable dry-run mode (Optional)
                - exclude_from_chat_history: Exclude from chat history (Optional)
                - exclude_context: Exclude context data (Optional)
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

                # NEW: Extract top-level configuration
                top_level_model = prompt_config.get('model')
                top_level_temperature = prompt_config.get('temperature')
                top_level_max_tokens = prompt_config.get('max_tokens')
                dry_run = prompt_config.get('dry_run', False)
                exclude_from_history = prompt_config.get('exclude_from_chat_history', False)
                exclude_context = prompt_config.get('exclude_context', False)

                # Get runtime params with defaults from kwargs, with top-level overrides
                temperature = top_level_temperature if top_level_temperature is not None else kwargs.get('temperature', 0)
                max_tokens = top_level_max_tokens if top_level_max_tokens is not None else kwargs.get('max_tokens', 8000)
                model = top_level_model if top_level_model is not None else getattr(client, 'model_name', None)
                
                # Initialize dry-run logger if needed
                if dry_run:
                    self.dry_run_logger = create_dry_run_logger()
                    logger.info(f"Dry-run mode enabled. Logs will be saved to: {self.dry_run_logger.dry_run_folder}")

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
                    model=model,
                    exclude_from_history=exclude_from_history,
                    exclude_context=exclude_context,
                    dry_run=dry_run,
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
                
                # Add configuration metadata
                result[name]['config_metadata'] = {
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'dry_run': dry_run,
                    'exclude_from_history': exclude_from_history,
                    'exclude_context': exclude_context
                }
                
                # Log summary in dry-run mode
                if dry_run and self.dry_run_logger:
                    self.dry_run_logger.log_pipeline_summary(
                        pipeline_name=name,
                        total_steps=1,  # Standard processor has 1 logical step
                        pipeline_config=prompt_config,
                        execution_metadata={
                            'has_schema': bool(schema),
                            'has_tonality': bool(tonality_messages),
                            'context_data_provided': bool(context_data),
                            'session_info': self.dry_run_logger.get_session_info()
                        }
                    )
                    
                    # NEW: Log the complete pipeline result
                    self.dry_run_logger.log_pipeline_result(
                        pipeline_name=name,
                        pipeline_result=result,
                        pipeline_config=prompt_config
                    )

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

                # Extract step-level configuration with fallbacks
                step_model = config.get('model')
                step_temperature = config.get('temperature', temperature)
                step_max_tokens = config.get('max_tokens', max_tokens)
                step_exclude_from_history = config.get('exclude_from_chat_history', False)
                step_exclude_context = config.get('exclude_context', False)
                step_dry_run = config.get('dry_run', False)
                
                # Process extraction and tonality with conversation history
                json_result, tonality_messages, tonality_result = self._process_extraction_with_tonality(
                    client=client,
                    extraction_prompt=config['prompt'],
                    json_schema=config.get('schema', None),
                    tonality_messages=config.get('tonality_messages', None),
                    context_data=context_data,
                    model=step_model,
                    exclude_from_history=step_exclude_from_history,
                    exclude_context=step_exclude_context,
                    dry_run=step_dry_run,
                    subsection_name=subsection_name,
                    conversation_id=conversation_id,
                    temperature=step_temperature,
                    max_tokens=step_max_tokens,
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
        model: str = None,
        exclude_from_history: bool = False,
        exclude_context: bool = False,
        dry_run: bool = False,
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
            logger.info(f"Model: {model if model else 'default'}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")
            logger.info(f"Dry run: {'Enabled' if dry_run else 'Disabled'}")
            logger.info(f"Exclude from history: {exclude_from_history}")
            logger.info(f"Exclude context: {exclude_context}")

            # Format context data as string if needed and not excluded
            if context_data is not None and not exclude_context:
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
                # If no context data or context excluded, use the prompt directly
                extraction_messages = [
                    {
                        "role": "system",
                        "content": extraction_prompt
                    }
                ]

            # Handle dry run mode for extraction
            if dry_run:
                logger.info(f"\n=== DRY RUN - Extraction Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {temperature}")
                logger.info(f"Max tokens: {max_tokens}")
                logger.info(f"Messages: {json.dumps(extraction_messages, indent=2)}")
                logger.info(f"Schema: {json.dumps(json_schema, indent=2) if json_schema else 'None'}")
                logger.info(f"Exclude from history: {exclude_from_history}")
                logger.info(f"Exclude context: {exclude_context}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    client_params = {
                        "messages": extraction_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "subsection_name": f"{subsection_name} (extraction)",
                        "conversation_id": conversation_id
                    }
                    
                    if json_schema:
                        client_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": json_schema
                        }
                        client_params["subsection_name"] = f"{subsection_name} (JSON extraction)"
                    
                    # Add model information
                    request_data = {
                        "model": model or 'default',
                        **client_params
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "context_data_included": not exclude_context and bool(context_data),
                        "has_schema": bool(json_schema),
                        "step_type": "extraction"
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=subsection_name,
                        step_name="extraction",
                        step_type="json_extraction",
                        request_data=request_data,
                        metadata=metadata
                    )
                
                extracted_json = {"dry_run": True, "message": "This is a dry run extraction response"}
            else:
                # Prepare client parameters
                client_params = {
                    "messages": extraction_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "subsection_name": f"{subsection_name} (extraction)",
                    "conversation_id": conversation_id
                }
                
                # Add any additional kwargs
                client_params.update(kwargs)
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    original_model = client.model_name
                    client.model_name = model
                
                # Run extraction with schema
                if json_schema:
                    client_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                    client_params["subsection_name"] = f"{subsection_name} (JSON extraction)"
                
                extracted_json = client.generate_completion(**client_params)
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

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

            # Handle dry run mode for tonality
            if dry_run:
                logger.info(f"\n=== DRY RUN - Tonality Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {temperature}")
                logger.info(f"Max tokens: {max_tokens}")
                logger.info(f"Messages: {json.dumps(processed_tonality_messages, indent=2)}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    tonality_params = {
                        "messages": processed_tonality_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "subsection_name": f"{subsection_name} (tonality matching)",
                        "conversation_id": f"{conversation_id}_tonality"
                    }
                    
                    # Add model information
                    request_data = {
                        "model": model or 'default',
                        **tonality_params
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "step_type": "tonality_matching",
                        "extracted_json_length": len(str(extracted_json))
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=subsection_name,
                        step_name="tonality_matching",
                        step_type="tonality_transformation",
                        request_data=request_data,
                        metadata=metadata
                    )
                
                tonality_result = "This is a dry run tonality response"
            else:
                # Prepare client parameters for tonality
                tonality_params = {
                    "messages": processed_tonality_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "subsection_name": f"{subsection_name} (tonality matching)",
                    "conversation_id": f"{conversation_id}_tonality"
                }
                
                # Add any additional kwargs
                tonality_params.update(kwargs)
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    original_model = client.model_name
                    client.model_name = model
                
                # Process tonality using a different conversation ID to avoid mixing contexts
                tonality_result = client.generate_completion(**tonality_params)
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

            return extracted_json, processed_tonality_messages, tonality_result

        except Exception as e:
            logger.error(
                f"Error in extraction and tonality for '{subsection_name}': {str(e)}")

            # Return placeholders instead of raising an exception
            placeholder_json = {
                "status": "failed", "error": f"Failed processing for '{subsection_name}'"}
            placeholder_tonality = f"[Content generation failed for '{subsection_name}']"

            return placeholder_json, [], placeholder_tonality
