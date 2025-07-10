# processors/cot_processor.py
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union

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


class ChainOfThoughtProcessor(BasePromptProcessor):
    """
    Enhanced Processor for Chain of Thought (CoT) pipelines with fine-tuning capabilities.

    New Feature: Automatic prompt fine-tuning using the 'fine_tune_prompt' keyword.
    When 'fine_tune_prompt' is present in the pipeline config, each step's prompt
    will be automatically enhanced before execution.
    """

    def __init__(self):
        """Initialize the Chain of Thought processor with fine-tuning capabilities"""
        self.message_history = []  # For CoT pipeline
        self.raw_history = []      # For CoT pipeline

        # Fine-tuning related attributes
        self.fine_tune_cache = {}  # Cache for fine-tuned prompts
        self.fine_tuning_enabled = False
        self.fine_tune_prompt = None
        self.prompt_enhancer = None
        self.pipeline_context = {}  # Store context from previous steps
        
        # Dry-run logging
        self.dry_run_logger = None

    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a Chain of Thought pipeline with optional fine-tuning"""
        try:
            # Extract pipeline configuration
            pipeline_name = prompt_config.get('name', 'unnamed_pipeline')
            steps = prompt_config.get('steps', [])
            context_data = prompt_config.get('context_data')

            # NEW: Extract top-level model configuration
            top_level_model = prompt_config.get('model')
            top_level_temperature = prompt_config.get('temperature')
            top_level_max_tokens = prompt_config.get('max_tokens')
            dry_run = prompt_config.get('dry_run', False)

            # NEW: Check for fine-tuning configuration
            self.fine_tune_prompt = prompt_config.get('fine_tune_prompt')
            
            # Validate fine_tune_prompt if provided
            if self.fine_tune_prompt is not None:
                if not self.fine_tune_prompt.strip():
                    logger.warning(
                        "⚠️ Fine-tuning prompt is empty or contains only whitespace. "
                        "Fine-tuning will be disabled."
                    )
                    self.fine_tune_prompt = None
            
            self.fine_tuning_enabled = bool(self.fine_tune_prompt and self.fine_tune_prompt.strip())
            
            # Initialize dry-run logger if needed (or if fine-tuning is enabled)
            if dry_run or self.fine_tuning_enabled:
                self.dry_run_logger = create_dry_run_logger()
                if dry_run:
                    logger.info(f"Dry-run mode enabled. Logs will be saved to: {self.dry_run_logger.dry_run_folder}")
                elif self.fine_tuning_enabled:
                    logger.info(f"Fine-tuning logging enabled. Logs will be saved to: {self.dry_run_logger.dry_run_folder}")

            if self.fine_tuning_enabled:
                logger.info(
                    f"🔧 Fine-tuning enabled for pipeline: {pipeline_name}")
                logger.info(
                    f"Fine-tuning instruction: {self.fine_tune_prompt[:100]}...")

                # Initialize the prompt enhancer
                self._initialize_prompt_enhancer(client, dry_run=dry_run)
            else:
                logger.info(f"📋 Processing standard pipeline: {pipeline_name}")

            # Validate required inputs
            if not steps:
                raise ValueError(
                    f"Missing required 'steps' in pipeline configuration for '{pipeline_name}'")

            # Reset history for this new pipeline
            self.message_history = []
            self.raw_history = []
            self.pipeline_context = {
                'pipeline_name': pipeline_name,
                'context_data': context_data,
                'completed_steps': []
            }

            # Store pipeline info for dry-run logging
            if dry_run and self.dry_run_logger:
                self._current_pipeline_name = pipeline_name
            
            # Process the pipeline
            pipeline_result = self._process_pipeline(
                client=client,
                pipeline_name=pipeline_name,
                steps=steps,
                context_data=context_data,
                top_level_model=top_level_model,
                top_level_temperature=top_level_temperature,
                top_level_max_tokens=top_level_max_tokens,
                dry_run=dry_run,
                **kwargs
            )

            # Add fine-tuning metadata to results
            if self.fine_tuning_enabled:
                pipeline_result['fine_tuning_applied'] = True
                pipeline_result['fine_tune_instruction'] = self.fine_tune_prompt
                pipeline_result['cache_hits'] = len(
                    [k for k in self.fine_tune_cache.keys()])
            
            # Add pipeline configuration metadata
            pipeline_result['pipeline_config'] = {
                'dry_run': dry_run,
                'top_level_model': top_level_model,
                'top_level_temperature': top_level_temperature,
                'top_level_max_tokens': top_level_max_tokens
            }
            
            # Log pipeline summary in dry-run mode
            if dry_run and self.dry_run_logger:
                self.dry_run_logger.log_pipeline_summary(
                    pipeline_name=pipeline_name,
                    total_steps=len(steps),
                    pipeline_config=prompt_config,
                    execution_metadata={
                        'fine_tuning_enabled': self.fine_tuning_enabled,
                        'context_data_provided': bool(context_data),
                        'total_failed_steps': len(pipeline_result.get('failed_steps', [])),
                        'session_info': self.dry_run_logger.get_session_info()
                    }
                )
                
                # NEW: Log the complete pipeline result
                complete_result = {pipeline_name: pipeline_result}
                self.dry_run_logger.log_pipeline_result(
                    pipeline_name=pipeline_name,
                    pipeline_result=complete_result,
                    pipeline_config=prompt_config
                )

            return {
                pipeline_name: pipeline_result
            }

        except Exception as e:
            logger.error(
                f"Error processing Chain of Thought pipeline: {str(e)}")
            return {
                'error': {
                    'success': False,
                    'message': f"Error processing Chain of Thought pipeline: {str(e)}"
                }
            }

    def _initialize_prompt_enhancer(self, client, dry_run=False):
        """Initialize the prompt enhancer for fine-tuning"""
        try:
            from llm_factory.utils import PromptEnhancer
            
            # Let PromptEnhancer auto-detect client type when client is provided
            # Only use default_client_type when no client is provided
            if client:
                self.prompt_enhancer = PromptEnhancer(
                    client=client,
                    dry_run=dry_run
                )
            else:
                from llm_factory.env_loader import ENV_VARS
                default_client_type = ENV_VARS.get('default_client_type', 'azure')
                self.prompt_enhancer = PromptEnhancer(
                    client_type=default_client_type,
                    dry_run=dry_run
                )
            logger.info("✅ Prompt enhancer initialized successfully")
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to initialize prompt enhancer: {str(e)}")
            self.fine_tuning_enabled = False

    def _fine_tune_step_prompt(self, step: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """
        Fine-tune a step's prompt using the PromptEnhancer.

        Args:
            step: The step configuration to fine-tune
            step_index: Index of the step in the pipeline

        Returns:
            Enhanced step configuration with fine-tuned prompt
        """
        if not self.fine_tuning_enabled or not self.prompt_enhancer:
            return step

        try:
            step_name = step.get('name', f'step_{step_index + 1}')
            original_prompt = step.get('prompt', '')

            if not original_prompt:
                logger.warning(
                    f"⚠️ No prompt to fine-tune in step: {step_name}")
                return step

            # Create cache key for this prompt
            cache_key = self._create_cache_key(original_prompt, step_index)

            # Check cache first
            cached_result = self._get_cached_fine_tuned_prompt(cache_key)
            if cached_result:
                logger.info(
                    f"🎯 Using cached fine-tuned prompt for step: {step_name}")
                
                # Log the cached fine-tuning process details (always log, regardless of dry-run)
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    self.dry_run_logger.log_fine_tuning_process(
                        pipeline_name=self.pipeline_context.get('pipeline_name', 'unknown'),
                        step_name=step_name,
                        original_prompt=original_prompt,
                        fine_tuning_guidelines=self.fine_tune_prompt,
                        enhanced_prompt=cached_result,
                        enhancement_metadata={
                            'step_index': step_index,
                            'cache_hit': True,
                            'enhancement_type': 'general',
                            'context_provided': bool(self._build_fine_tuning_context(step, step_index))
                        }
                    )
                
                enhanced_step = step.copy()
                enhanced_step['prompt'] = cached_result
                enhanced_step['_fine_tuned'] = True
                return enhanced_step

            logger.info(f"🔧 Fine-tuning prompt for step: {step_name}")

            # Build context for fine-tuning
            context_data = self._build_fine_tuning_context(step, step_index)

            # Fine-tune the prompt
            enhancement_result = self.prompt_enhancer.enhance_prompt(
                base_prompt=original_prompt,
                new_prompt=self.fine_tune_prompt,
                context_data=context_data,
                enhancement_type="general",
                temperature=0.3,
                max_tokens=4000,
                explain=False  # Don't need explanations for pipeline processing
            )

            if enhancement_result.get('enhanced_prompt'):
                enhanced_prompt = enhancement_result['enhanced_prompt']

                # Cache the result
                self._cache_fine_tuned_prompt(cache_key, enhanced_prompt)
                
                # Log the fine-tuning process details (always log, regardless of dry-run)
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    self.dry_run_logger.log_fine_tuning_process(
                        pipeline_name=self.pipeline_context.get('pipeline_name', 'unknown'),
                        step_name=step_name,
                        original_prompt=original_prompt,
                        fine_tuning_guidelines=self.fine_tune_prompt,
                        enhanced_prompt=enhanced_prompt,
                        enhancement_metadata={
                            'step_index': step_index,
                            'cache_hit': False,
                            'enhancement_type': 'general',
                            'context_provided': bool(context_data)
                        }
                    )

                # Create enhanced step
                enhanced_step = step.copy()
                enhanced_step['prompt'] = enhanced_prompt
                enhanced_step['_fine_tuned'] = True
                enhanced_step['_original_prompt'] = original_prompt

                logger.info(
                    f"✅ Successfully fine-tuned prompt for step: {step_name}")
                return enhanced_step
            else:
                logger.warning(
                    f"⚠️ Fine-tuning failed for step: {step_name}, using original prompt")
                return step

        except Exception as e:
            logger.error(f"❌ Error fine-tuning step {step_name}: {str(e)}")
            return step

    def _build_fine_tuning_context(self, step: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """Build context data for fine-tuning based on pipeline state and previous steps"""
        context = {
            'pipeline_info': {
                'name': self.pipeline_context.get('pipeline_name'),
                'current_step_index': step_index,
                'step_name': step.get('name'),
                'step_type': step.get('type')
            }
        }

        # Add pipeline context data if available
        if self.pipeline_context.get('context_data'):
            context['pipeline_context_data'] = self.pipeline_context['context_data']

        # Add previous step results for context
        completed_steps = self.pipeline_context.get('completed_steps', [])
        if completed_steps:
            context['previous_steps'] = []
            # Include last 2 steps for context (to avoid token limits)
            for prev_step in completed_steps[-2:]:
                step_summary = {
                    'step_name': prev_step.get('name'),
                    'step_type': prev_step.get('type'),
                    'output_summary': str(prev_step.get('output', ''))[:200] + '...' if len(str(prev_step.get('output', ''))) > 200 else str(prev_step.get('output', ''))
                }
                context['previous_steps'].append(step_summary)

        # Add schema information if present
        if step.get('schema'):
            context['expected_output_schema'] = step['schema']

        return context

    def _create_cache_key(self, prompt: str, step_index: int) -> str:
        """Create a unique cache key for a prompt and its context"""
        # Include fine_tune_prompt and step_index in the hash for uniqueness
        cache_content = f"{prompt}|{self.fine_tune_prompt}|{step_index}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _cache_fine_tuned_prompt(self, cache_key: str, enhanced_prompt: str):
        """Cache a fine-tuned prompt"""
        self.fine_tune_cache[cache_key] = enhanced_prompt

    def _get_cached_fine_tuned_prompt(self, cache_key: str) -> Optional[str]:
        """Retrieve a cached fine-tuned prompt"""
        return self.fine_tune_cache.get(cache_key)

    def _process_pipeline(
        self,
        client,
        pipeline_name: str,
        steps: List[Dict[str, Any]],
        context_data: Any = None,
        top_level_model: str = None,
        top_level_temperature: float = None,
        top_level_max_tokens: int = None,
        dry_run: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a multi-step Chain of Thought pipeline with fine-tuning support"""
        try:
            if not steps:
                return {
                    'success': False,
                    'error': 'No steps defined in pipeline configuration',
                    'final_output': None
                }

            # Get parameters from kwargs with defaults, with top-level overrides
            temperature = top_level_temperature if top_level_temperature is not None else kwargs.get('temperature', 0)
            max_tokens = top_level_max_tokens if top_level_max_tokens is not None else kwargs.get('max_tokens', 8000)
            model = top_level_model if top_level_model is not None else getattr(client, 'model_name', None)
            logger.info(f"\n=== Processing Pipeline: {pipeline_name} ===")
            logger.info(f"Total steps: {len(steps)}")
            logger.info(
                f"Fine-tuning: {'Enabled' if self.fine_tuning_enabled else 'Disabled'}")
            logger.info(f"Dry run: {'Enabled' if dry_run else 'Disabled'}")
            logger.info(f"Model: {model if model else 'Default'}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")

            # Improved context data logging - check both pipeline and step level
            first_step_context = steps[0].get(
                'context_data') if steps else None
            has_context = context_data is not None or first_step_context is not None
            logger.info(
                f"Initial context data: {'Provided' if has_context else 'None'}")

            results = {}
            failed_steps = []


            for i, step in enumerate(steps):
                step_name = step.get('name', f"step_{i+1}")
                step_type = step.get('type', 'unknown')
                
                # Store current step index for dry-run logging
                if dry_run and self.dry_run_logger:
                    self._current_step_index = i

                # Add context to first step if provided
                if i == 0 and context_data is not None:
                    step['context_data'] = context_data

                # NEW: Fine-tune the step prompt if fine-tuning is enabled
                processed_step = step
                if self.fine_tuning_enabled:
                    processed_step = self._fine_tune_step_prompt(step, i)
                    if processed_step.get('_fine_tuned'):
                        logger.info(
                            f"🎯 Using fine-tuned prompt for step: {step_name}")

                # Print step configuration details
                logger.info(f"\n🚀 Processing step {i+1}/{len(steps)}:")
                logger.info(f"Name: {step_name}")
                logger.info(f"Type: {step_type}")
                if processed_step.get('_fine_tuned'):
                    logger.info("🔧 Fine-tuned: Yes")

                try:
                    # Get step-level parameters with fallbacks
                    step_model = processed_step.get('model', model)
                    step_temperature = processed_step.get('temperature', temperature)
                    step_max_tokens = processed_step.get('max_tokens', max_tokens)
                    step_exclude_from_history = processed_step.get('exclude_from_chat_history', False)
                    step_exclude_context = processed_step.get('exclude_context', False)
                    
                    # Process step based on type using the processed (potentially fine-tuned) step
                    if processed_step.get('messages') or step_type.lower() == 'tonality':
                        step_result = self._process_tonality_step(
                            client, processed_step, results, step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    elif step_type.lower() in ['initial', 'initialprompt']:
                        step_result = self._process_example_step(
                            client, processed_step, results, step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    elif step_type.lower() in ['newproblem', 'newquestion']:
                        step_result = self._process_new_problem_step(
                            client, processed_step, results, step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    elif step_type.lower() in ['followup', 'history']:
                        step_result = self._process_regular_step(
                            client, processed_step, results, "FOLLOWUP", step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    elif step_type.lower() in ['verification', 'verify']:
                        step_result = self._process_verification_step(
                            client, processed_step, results, step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    elif step_type.lower() in ['summary', 'summarize']:
                        step_result = self._process_summary_step(
                            client, processed_step, results, step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                    else:
                        step_result = self._process_regular_step(
                            client, processed_step, results, "QUESTION", step_temperature, step_max_tokens,
                            step_model, step_exclude_from_history, step_exclude_context, dry_run)
                except Exception as e:
                    step_result = {
                        'success': False,
                        'error': f"Unexpected error processing step: {str(e)}"
                    }

                # Handle step results and update pipeline context
                if step_result['success']:
                    output_key = processed_step.get(
                        'output_key', f"{step_name}_output")
                    results[output_key] = step_result['output']

                    # Update pipeline context with completed step
                    self.pipeline_context['completed_steps'].append({
                        'name': step_name,
                        'type': step_type,
                        'output': step_result['output'],
                        'fine_tuned': processed_step.get('_fine_tuned', False)
                    })

                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    error_msg = step_result.get('error', 'Unknown error')
                    failed_steps.append({
                        'step_name': step_name,
                        'error': error_msg
                    })
                    logger.info(f"❌ {step_name} failed: {error_msg}")

            # Final pipeline status
            if failed_steps:
                logger.info(
                    f"\n⚠️ Pipeline completed with {len(failed_steps)} failed steps")
                return {
                    'success': False,
                    'error': f"Failed {len(failed_steps)} steps",
                    'failed_steps': failed_steps,
                    'partial_results': results,
                    'history_messages': self.raw_history,
                    'final_output': "Pipeline failed to complete successfully"
                }

            logger.info(f"\n✅ Pipeline completed successfully")
            if self.fine_tuning_enabled:
                logger.info(
                    f"🔧 Fine-tuning cache size: {len(self.fine_tune_cache)} entries")

            # Get the last step's output key for the final result
            last_output_key = None
            if steps:
                last_step = steps[-1]
                last_output_key = last_step.get(
                    'output_key', f"{last_step.get('name', 'final')}_output")

            # If we have JSON schema in an earlier step, use that as json_result
            json_result = None
            for step in steps:
                if step.get('schema') and step.get('output_key') in results:
                    # Found a step with schema - use its output as the json_result
                    output_key = step.get('output_key')
                    json_result = results[output_key]
                    break

            # If no step with schema was found, use the first step result as json_result
            if json_result is None and len(results) > 0:
                first_output_key = steps[0].get(
                    'output_key', f"{steps[0].get('name', 'first')}_output")
                if first_output_key in results:
                    json_result = results[first_output_key]

            # Get final output from the last step if available
            final_output = None
            if last_output_key and last_output_key in results:
                final_output = results[last_output_key]
            elif results:
                # If we can't find the exact key, use the last result added
                final_output = next(iter(results.values()))

            return {
                'success': True,
                'steps': results,
                'conversation_history': self._format_history(),
                'json_result': json_result,
                'tonality_result': results.get(last_output_key, "No final output available") if last_output_key else "No final output available",
                'final_output': final_output if final_output is not None else "No final output available",
                'history_messages': self.raw_history
            }

        except Exception as e:
            error_msg = f"Pipeline processing error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'partial_results': results if 'results' in locals() else {},
                'history_messages': self.raw_history if hasattr(self, 'raw_history') else [],
                'final_output': f"Error: {error_msg}"
            }

    # [All the existing methods remain unchanged - just keeping the important ones for context]

    def _format_history(self) -> str:
        """Format the conversation history for display"""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        formatted = "=== CONVERSATION HISTORY ===\n\n"

        for i, msg in enumerate(self.message_history):
            if isinstance(msg, SystemMessage):
                formatted += f"SYSTEM [{i}]: {msg.content[:100]}...\n\n"
            elif isinstance(msg, HumanMessage):
                formatted += f"HUMAN [{i}]: {msg.content[:200]}...\n\n"
            elif isinstance(msg, AIMessage):
                formatted += f"AI [{i}]: {msg.content[:200]}...\n\n"
            else:
                formatted += f"UNKNOWN [{i}]: {str(msg)[:100]}...\n\n"

        return formatted

    def _process_example_step(self, client, step, previous_results, temperature, max_tokens, 
                             model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a step that establishes a pattern with examples (with schema support)"""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        try:
            prompt_text = step.get('prompt', '')
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)
            schema = step.get('schema')

            if not prompt_text:
                return {
                    'success': False,
                    'error': 'Empty prompt in example step'
                }

            system_msg = SystemMessage(content=(
                "You are an AI assistant that uses chain-of-thought reasoning to solve problems. "
                "Below are examples of the reasoning pattern to follow. "
                "For new problems, follow the same step-by-step approach."
            ))

            user_content = "### EXAMPLES OF REASONING PATTERN:\n\n" + prompt_text
            if context_data and not exclude_context:
                context_str = context_data if isinstance(
                    context_data, str) else json.dumps(context_data, indent=2)
                user_content += f"\n\n### CONTEXT DATA TO USE:\n\n{context_str}"

            if schema:
                schema_str = json.dumps(schema, indent=2)
                user_content += (
                    f"\n\nYou MUST format your response as a valid JSON object conforming to the following schema:\n"
                    f"```json\n{schema_str}\n```\n"
                    f"Important: Your response must be a properly formatted JSON object without any additional text or explanation."
                )

            user_msg = HumanMessage(content=user_content)

            # Only add to history if not excluded
            if not exclude_from_history:
                self.message_history.append(system_msg)
                self.message_history.append(user_msg)

                self.raw_history.extend([
                    {'role': 'system', 'content': system_msg.content},
                    {'role': 'user', 'content': user_msg.content}
                ])

            messages = [
                {"role": "system", "content": system_msg.content},
                {"role": "user", "content": user_msg.content}
            ]

            # Handle dry run mode
            if dry_run:
                logger.info(f"\n=== DRY RUN - Step Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {step_temperature}")
                logger.info(f"Max tokens: {step_max_tokens}")
                logger.info(f"Messages: {json.dumps(messages, indent=2)}")
                logger.info(f"Schema: {json.dumps(schema, indent=2) if schema else 'None'}")
                logger.info(f"Exclude from history: {exclude_from_history}")
                logger.info(f"Exclude context: {exclude_context}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    request_data = {
                        "model": model or 'default',
                        "temperature": step_temperature,
                        "max_tokens": step_max_tokens,
                        "messages": messages,
                        "response_format": {"type": "json_schema", "json_schema": schema} if schema else None
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "context_data_included": not exclude_context and bool(context_data),
                        "step_index": getattr(self, '_current_step_index', 0),
                        "fine_tuned": step.get('_fine_tuned', False)
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=getattr(self, '_current_pipeline_name', 'unknown'),
                        step_name=step.get('name', 'unknown_step'),
                        step_type=step.get('type', 'example_step'),
                        request_data=request_data,
                        metadata=metadata
                    )
                
                # Return mock response for dry run
                ai_response = {"dry_run": True, "message": "This is a dry run response"}
            else:
                # Prepare client parameters
                client_params = {
                    "messages": messages,
                    "temperature": step_temperature,
                    "max_tokens": step_max_tokens
                }
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    # Temporarily update client model for this request
                    original_model = client.model_name
                    client.model_name = model
                    
                if schema:
                    client_params["response_format"] = {
                        "type": "json_schema", "json_schema": schema
                    }
                
                # Add fine-tuning context if enabled
                if self.fine_tuning_enabled:
                    client_params["fine_tune_context"] = {
                        "enabled": True,
                        "prompt": self.fine_tune_prompt
                    }
                
                ai_response = client.generate_completion(**client_params)
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

            ai_msg = AIMessage(content=str(ai_response) if isinstance(
                ai_response, dict) else ai_response)
            
            # Only add to history if not excluded
            if not exclude_from_history:
                self.message_history.append(ai_msg)

                self.raw_history.append({
                    'role': 'assistant',
                    'content': ai_response if isinstance(ai_response, str) else json.dumps(ai_response, indent=2)
                })

            return {
                'success': True,
                'output': ai_response
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing example step: {str(e)}"
            }

    def _process_new_problem_step(self, client, step, previous_results, temperature, max_tokens,
                                 model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a new problem step with improved schema handling"""
        from langchain_core.messages import HumanMessage, AIMessage
        try:
            # Prune history to keep it manageable
            self._prune_message_history(max_messages=5)

            # Get step parameters
            prompt_text = step.get('prompt', '')
            input_key = step.get('input_key', '')
            schema = step.get('schema')
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)

            if not prompt_text:
                return {'success': False, 'error': 'Empty prompt in new problem step'}

            # Build the prompt by including previous results if specified
            if input_key and input_key in previous_results:
                prev_result = previous_results[input_key]
                # Extract just the conclusion for brevity
                conclusion = self._extract_conclusion(prev_result)
                prompt_text = f"Previous result conclusion:\n```\n{conclusion}\n```\n\n{prompt_text}"

            if "previous analysis" not in prompt_text.lower():
                prompt_text = f"Continue the analysis based on the previous conclusion. Be concise.\n\n{prompt_text}"

            # Add context data if provided and not excluded
            if context_data and not exclude_context:
                context_str = context_data if isinstance(
                    context_data, str) else json.dumps(context_data, indent=2)
                prompt_text = f"Context Data:\n```\n{context_str}\n```\n\n{prompt_text}"

            # Enhance prompt for JSON schema if present
            if schema:
                # If schema is a string, try to parse it as JSON
                if isinstance(schema, str):
                    try:
                        schema = json.loads(schema)
                    except:
                        pass

                # Add explicit instructions about the schema format
                schema_str = json.dumps(schema, indent=2)
                prompt_text += f"\n\nYou MUST format your response as a valid JSON object that conforms to the following schema:\n```json\n{schema_str}\n```\n"
                prompt_text += "Important: Your response must be a properly formatted JSON object without any additional text, markdown, or explanation."

            # Create and add user message to history (if not excluded)
            user_msg = HumanMessage(content=prompt_text)
            if not exclude_from_history:
                self.message_history.append(user_msg)
                self.raw_history.append(
                    {'role': 'user', 'content': user_msg.content})

            # Convert message history to API format
            messages = []
            for msg in self.message_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append(
                        {"role": "assistant", "content": msg.content})
                else:
                    messages.append({"role": "system", "content": msg.content})

            # Handle dry run mode
            if dry_run:
                logger.info(f"\n=== DRY RUN - Step Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {step_temperature}")
                logger.info(f"Max tokens: {step_max_tokens}")
                logger.info(f"Messages: {json.dumps(messages, indent=2)}")
                logger.info(f"Schema: {json.dumps(schema, indent=2) if schema else 'None'}")
                logger.info(f"Exclude from history: {exclude_from_history}")
                logger.info(f"Exclude context: {exclude_context}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    request_data = {
                        "model": model or 'default',
                        "temperature": step_temperature,
                        "max_tokens": step_max_tokens,
                        "messages": messages,
                        "response_format": {"type": "json_schema", "json_schema": schema} if schema else None
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "context_data_included": not exclude_context and bool(context_data),
                        "step_index": getattr(self, '_current_step_index', 0),
                        "fine_tuned": step.get('_fine_tuned', False)
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=getattr(self, '_current_pipeline_name', 'unknown'),
                        step_name=step.get('name', 'unknown_step'),
                        step_type=step.get('type', 'new_problem_step'),
                        request_data=request_data,
                        metadata=metadata
                    )
                
                response = {"dry_run": True, "message": "This is a dry run response"}
                structured_output = response
            else:
                # Prepare client parameters
                client_params = {
                    "messages": messages,
                    "temperature": step_temperature,
                    "max_tokens": step_max_tokens
                }
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    original_model = client.model_name
                    client.model_name = model
                
                # Generate response using the client
                if schema:
                    client_params["response_format"] = {
                        "type": "json_schema", "json_schema": schema
                    }
                
                # Add fine-tuning context if enabled
                if self.fine_tuning_enabled:
                    client_params["fine_tune_context"] = {
                        "enabled": True,
                        "prompt": self.fine_tune_prompt
                    }
                
                response = client.generate_completion(**client_params)
                structured_output = response
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

            # Add AI response to history (if not excluded)
            ai_msg = AIMessage(content=str(response) if isinstance(
                response, dict) else response)
            
            if not exclude_from_history:
                self.message_history.append(ai_msg)

                # Use the actual content for the raw history
                self.raw_history.append({
                    'role': 'assistant',
                    'content': response if isinstance(response, str) else json.dumps(response, indent=2)
                })

            return {'success': True, 'output': structured_output}

        except Exception as e:
            return {'success': False, 'error': f"Error processing new problem step: {str(e)}"}

    def _process_tonality_step(self, client, step, previous_results, temperature, max_tokens,
                              model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a tonality step"""
        try:
            # Get step parameters
            input_key = step.get('input_key', '')
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)

            # Get input value which can be either a list of message objects or a string
            messages_input = step.get('messages', [])
            prompt_text = step.get('prompt', '')

            # Check if we have either messages or a prompt
            if not messages_input and not prompt_text:
                return {
                    'success': False,
                    'error': 'No messages or prompt provided for tonality step'
                }

            # Get the input data from previous results
            if not input_key or input_key not in previous_results:
                return {
                    'success': False,
                    'error': f"Missing required input_key '{input_key}' for tonality step"
                }

            input_data = previous_results[input_key]

            # Convert input data to string if it's not already
            if isinstance(input_data, dict) or isinstance(input_data, list):
                input_str = json.dumps(input_data, indent=2)
            else:
                input_str = str(input_data)

            # Create message list for the API call
            api_messages = []

            # Determine whether we're using a message list or a prompt string
            if isinstance(messages_input, list) and len(messages_input) > 0:
                # Process the messages list - each message should have 'role' and 'content'
                for i, msg in enumerate(messages_input):
                    # Skip the last message if it's from the user (we'll add our own)
                    if i == len(messages_input) - 1 and msg.get('role') == 'user':
                        continue

                    role = msg.get('role', 'user')
                    content = msg.get('content', '')

                    api_messages.append({
                        "role": role,
                        "content": content
                    })
            else:
                # Use the prompt text as a system message
                api_messages.append({
                    "role": "system",
                    "content": prompt_text
                })

            # Add context data to the user message if available and not excluded
            user_content = f"Convert the following drug information to a standardized list:\n{input_str}"
            if context_data and not exclude_context:
                if isinstance(context_data, str):
                    context_str = context_data
                else:
                    context_str = json.dumps(context_data, indent=2)

                # Add context data to the user message
                user_content = (
                    f"Use the following context data for this task:\n\n"
                    f"```\n{context_str}\n```\n\n"
                    f"{user_content}"
                )

            # Add our user message with the input data
            api_messages.append({
                "role": "user",
                "content": user_content
            })

            # Handle dry run mode
            if dry_run:
                logger.info(f"\n=== DRY RUN - Tonality Step Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {step_temperature}")
                logger.info(f"Max tokens: {step_max_tokens}")
                logger.info(f"Messages: {json.dumps(api_messages, indent=2)}")
                logger.info(f"Exclude from history: {exclude_from_history}")
                logger.info(f"Exclude context: {exclude_context}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    request_data = {
                        "model": model or 'default',
                        "temperature": step_temperature,
                        "max_tokens": step_max_tokens,
                        "messages": api_messages
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "context_data_included": not exclude_context and bool(context_data),
                        "step_index": getattr(self, '_current_step_index', 0),
                        "fine_tuned": step.get('_fine_tuned', False),
                        "input_key": step.get('input_key')
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=getattr(self, '_current_pipeline_name', 'unknown'),
                        step_name=step.get('name', 'unknown_step'),
                        step_type=step.get('type', 'tonality_step'),
                        request_data=request_data,
                        metadata=metadata
                    )
                
                ai_response = "This is a dry run tonality response"
            else:
                # Prepare client parameters
                client_params = {
                    "messages": api_messages,
                    "temperature": step_temperature,
                    "max_tokens": step_max_tokens
                }
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    original_model = client.model_name
                    client.model_name = model
                
                # Add fine-tuning context if enabled
                if self.fine_tuning_enabled:
                    client_params["fine_tune_context"] = {
                        "enabled": True,
                        "prompt": self.fine_tune_prompt
                    }
                
                # Generate response using the client
                ai_response = client.generate_completion(**client_params)
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

            # Add a simplified version to history to maintain continuity (if not excluded)
            if not exclude_from_history:
                from langchain_core.messages import HumanMessage, AIMessage
                history_user_msg = HumanMessage(
                    content="Please format the previous information according to our style guidelines.")
                self.message_history.append(history_user_msg)
                ai_msg = AIMessage(content=ai_response)
                self.message_history.append(ai_msg)

                # Track in raw history
                self.raw_history.append({
                    'role': 'user',
                    'content': history_user_msg.content
                })
                self.raw_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })

            return {
                'success': True,
                'output': ai_response
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing tonality step: {str(e)}"
            }

    def _process_regular_step(self, client, step, previous_results, step_label="QUESTION", temperature=0, max_tokens=8000,
                             model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a regular follow-up step or question with schema support"""
        from langchain_core.messages import HumanMessage, AIMessage

        try:
            # Get step parameters
            prompt_text = step.get('prompt', '')
            input_key = step.get('input_key', '')
            schema = step.get('schema')  # Add schema parameter
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)

            if not prompt_text:
                return {
                    'success': False,
                    'error': f'Empty prompt in {step_label.lower()} step'
                }

            # If this step references a previous result, incorporate it into the prompt
            modified_prompt = prompt_text
            if input_key and input_key in previous_results:
                previous_result = previous_results[input_key]
                logger.info(
                    f"Using previous result from '{input_key}' for step")

                # Extract just the conclusion rather than full previous result
                conclusion = self._extract_conclusion(previous_result)

                # Incorporate previous result into the prompt
                modified_prompt = (
                    f"Based on previous conclusion:\n\n"
                    f"```\n{conclusion}\n```\n\n"
                    f"{prompt_text}"
                )

            # Add context data to the prompt if available and not excluded
            if context_data and not exclude_context:
                if isinstance(context_data, str):
                    context_str = context_data
                else:
                    context_str = json.dumps(context_data, indent=2)

                # Modify the prompt to include the context data
                modified_prompt = (
                    f"Use the following context data for this task:\n\n"
                    f"```\n{context_str}\n```\n\n"
                    f"{modified_prompt}"
                )

            # Add schema information if provided
            if schema:
                schema_str = json.dumps(schema, indent=2)
                modified_prompt += (
                    f"\n\nYou MUST format your response as a valid JSON object that conforms "
                    f"to the following schema:\n```json\n{schema_str}\n```\n"
                    f"Important: Your response must be a properly formatted JSON object "
                    f"without any additional text, markdown, or explanation."
                )

            # Create user message
            user_msg = HumanMessage(content=f"### {step_label}:\n\n" + modified_prompt
                                    )

            # Add message to history (if not excluded)
            if not exclude_from_history:
                self.message_history.append(user_msg)

                # Track raw history for debugging
                self.raw_history.append({
                    'role': 'user',
                    'content': user_msg.content
                })

            # Convert the message history to a format the client can use
            messages = []
            for msg in self.message_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append(
                        {"role": "assistant", "content": msg.content})
                else:
                    messages.append({"role": "system", "content": msg.content})

            # Handle dry run mode
            if dry_run:
                logger.info(f"\n=== DRY RUN - Step Payload ===")
                logger.info(f"Model: {model or 'default'}")
                logger.info(f"Temperature: {step_temperature}")
                logger.info(f"Max tokens: {step_max_tokens}")
                logger.info(f"Messages: {json.dumps(messages, indent=2)}")
                logger.info(f"Schema: {json.dumps(schema, indent=2) if schema else 'None'}")
                logger.info(f"Exclude from history: {exclude_from_history}")
                logger.info(f"Exclude context: {exclude_context}")
                logger.info(f"=== END DRY RUN ===")
                
                # Log to file if logger is available
                if hasattr(self, 'dry_run_logger') and self.dry_run_logger:
                    request_data = {
                        "model": model or 'default',
                        "temperature": step_temperature,
                        "max_tokens": step_max_tokens,
                        "messages": messages,
                        "response_format": {"type": "json_schema", "json_schema": schema} if schema else None
                    }
                    
                    metadata = {
                        "exclude_from_history": exclude_from_history,
                        "exclude_context": exclude_context,
                        "context_data_included": not exclude_context and bool(context_data),
                        "step_index": getattr(self, '_current_step_index', 0),
                        "fine_tuned": step.get('_fine_tuned', False)
                    }
                    
                    self.dry_run_logger.log_request(
                        pipeline_name=getattr(self, '_current_pipeline_name', 'unknown'),
                        step_name=step.get('name', 'unknown_step'),
                        step_type=step.get('type', 'regular_step'),
                        request_data=request_data,
                        metadata=metadata
                    )
                
                ai_response = {"dry_run": True, "message": "This is a dry run response"}
            else:
                # Prepare client parameters
                client_params = {
                    "messages": messages,
                    "temperature": step_temperature,
                    "max_tokens": step_max_tokens
                }
                
                # Add model if specified
                if model and hasattr(client, 'model_name'):
                    original_model = client.model_name
                    client.model_name = model
                
                # Generate response using the client
                if schema:
                    client_params["response_format"] = {
                        "type": "json_schema", "json_schema": schema
                    }
                
                # Add fine-tuning context if enabled
                if self.fine_tuning_enabled:
                    client_params["fine_tune_context"] = {
                        "enabled": True,
                        "prompt": self.fine_tune_prompt
                    }
                
                ai_response = client.generate_completion(**client_params)
                
                # Restore original model
                if model and hasattr(client, 'model_name'):
                    client.model_name = original_model

            # Add AI response to history (if not excluded)
            ai_msg = AIMessage(content=str(ai_response) if isinstance(
                ai_response, dict) else ai_response)
            
            if not exclude_from_history:
                self.message_history.append(ai_msg)

                # Track raw history for debugging - use the actual response
                self.raw_history.append({
                    'role': 'assistant',
                    'content': ai_response if isinstance(ai_response, str) else json.dumps(ai_response, indent=2)
                })

            return {
                'success': True,
                'output': ai_response
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing {step_label.lower()} step: {str(e)}"
            }

    def _process_verification_step(self, client, step, previous_results, temperature=0, max_tokens=8000,
                                  model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a verification step"""
        return self._process_regular_step(client, step, previous_results, "VERIFICATION CHECK", temperature, max_tokens,
                                        model, exclude_from_history, exclude_context, dry_run)

    def _process_summary_step(self, client, step, previous_results, temperature=0, max_tokens=8000,
                             model=None, exclude_from_history=False, exclude_context=False, dry_run=False):
        """Process a summary step"""
        return self._process_regular_step(client, step, previous_results, "SUMMARY", temperature, max_tokens,
                                        model, exclude_from_history, exclude_context, dry_run)

    def _extract_conclusion(self, text):
        """Extract just the conclusion from a longer text"""
        # For math problems, look for lines with solutions/coordinate pairs
        if isinstance(text, str):
            lines = text.split('\n')
            # Try to find lines with solution patterns
            for pattern in ["solution", "answer", "values", "x", "y", "=", "conclusion"]:
                for line in reversed(lines):  # Check from the end
                    if pattern.lower() in line.lower():
                        return line.strip()

            # If no clear conclusion line found, just take the last 2-3 lines
            return "\n".join(lines[-3:])
        return str(text)

    def _prune_message_history(self, max_messages=5):
        """Prune message history to keep only important context"""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        if len(self.message_history) <= max_messages*2:
            return  # No need to prune if we're under the limit

        # Keep system messages and recent conversation pairs
        system_messages = [
            msg for msg in self.message_history if isinstance(msg, SystemMessage)]
        other_messages = [
            msg for msg in self.message_history if not isinstance(msg, SystemMessage)]

        # Keep the last max_messages pairs (human + AI)
        recent_messages = other_messages[-max_messages *
                                         2:] if other_messages else []

        # Reconstruct history
        self.message_history = system_messages + recent_messages

        # Also prune raw history
        if len(self.raw_history) > max_messages*2 + len(system_messages):
            system_entries = [
                entry for entry in self.raw_history if entry.get('role') == 'system']
            other_entries = [
                entry for entry in self.raw_history if entry.get('role') != 'system']
            recent_entries = other_entries[-max_messages *
                                           2:] if other_entries else []
            self.raw_history = system_entries + recent_entries
