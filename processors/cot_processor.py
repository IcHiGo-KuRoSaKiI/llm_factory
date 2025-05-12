
# processors/cot_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Union

from .base_processor import BasePromptProcessor

logger = logging.getLogger(__name__)

class ChainOfThoughtProcessor(BasePromptProcessor):
    """Processor for Chain of Thought (CoT) pipelines"""
    
    def __init__(self):
        """Initialize the Chain of Thought processor"""
        self.message_history = []  # For CoT pipeline
        self.raw_history = []      # For CoT pipeline
    
    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a Chain of Thought pipeline"""
        try:
            # Extract pipeline configuration
            pipeline_name = prompt_config.get('name', 'unnamed_pipeline')
            steps = prompt_config.get('steps', [])
            context_data = prompt_config.get('context_data')
            
            # Validate required inputs
            if not steps:
                raise ValueError(f"Missing required 'steps' in pipeline configuration for '{pipeline_name}'")
            
            # Reset history for this new pipeline
            self.message_history = []
            self.raw_history = []
            
            # Process the pipeline
            pipeline_result = self._process_pipeline(
                client=client,
                pipeline_name=pipeline_name,
                steps=steps,
                context_data=context_data,
                **kwargs
            )
            
            return {
                pipeline_name: pipeline_result
            }
            
        except Exception as e:
            logger.error(f"Error processing Chain of Thought pipeline: {str(e)}")
            return {
                'error': {
                    'success': False,
                    'message': f"Error processing Chain of Thought pipeline: {str(e)}"
                }
            }
    
    def _process_pipeline(
        self,
        client,
        pipeline_name: str,
        steps: List[Dict[str, Any]],
        context_data: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a multi-step Chain of Thought pipeline"""
        try:
            if not steps:
                return {
                    'success': False,
                    'error': 'No steps defined in pipeline configuration',
                    'final_output': None
                }
            
            logger.info(f"\n=== Processing Pipeline: {pipeline_name} ===")
            logger.info(f"Total steps: {len(steps)}")
            logger.info(f"Initial context data: {'Provided' if context_data else 'None'}")
            
            results = {}
            failed_steps = []
            
            # Get parameters from kwargs with defaults
            temperature = kwargs.get('temperature', 0)
            max_tokens = kwargs.get('max_tokens', 8000)

            for i, step in enumerate(steps):
                step_name = step.get('name', f"step_{i+1}")
                step_type = step.get('type', 'unknown')
                
                # Add context to first step if provided
                if i == 0 and context_data is not None:
                    step['context_data'] = context_data

                # Print step configuration details
                logger.info(f"\n🚀 Processing step {i+1}/{len(steps)}:")
                logger.info(f"Name: {step_name}")
                logger.info(f"Type: {step_type}")
                
                try:
                    # Process step based on type
                    if step.get('messages') or step_type.lower() == 'tonality':
                        step_result = self._process_tonality_step(client, step, results, temperature, max_tokens)
                    elif step_type.lower() in ['initial', 'initialprompt']:
                        step_result = self._process_example_step(client, step, results, temperature, max_tokens)
                    elif step_type.lower() in ['newproblem', 'newquestion']:
                        step_result = self._process_new_problem_step(client, step, results, temperature, max_tokens)
                    elif step_type.lower() in ['followup', 'history']:
                        step_result = self._process_regular_step(client, step, results, "FOLLOWUP", temperature, max_tokens)
                    elif step_type.lower() in ['verification', 'verify']:
                        step_result = self._process_verification_step(client, step, results, temperature, max_tokens)
                    elif step_type.lower() in ['summary', 'summarize']:
                        step_result = self._process_summary_step(client, step, results, temperature, max_tokens)
                    else:
                        step_result = self._process_regular_step(client, step, results, "QUESTION", temperature, max_tokens)
                except Exception as e:
                    step_result = {
                        'success': False,
                        'error': f"Unexpected error processing step: {str(e)}"
                    }

                # Handle step results
                if step_result['success']:
                    output_key = step.get('output_key', f"{step_name}_output")
                    results[output_key] = step_result['output']
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
                logger.info(f"\n⚠️ Pipeline completed with {len(failed_steps)} failed steps")
                return {
                    'success': False,
                    'error': f"Failed {len(failed_steps)} steps",
                    'failed_steps': failed_steps,
                    'partial_results': results,
                    'history_messages': self.raw_history,
                    'final_output': "Pipeline failed to complete successfully"
                }
            
            logger.info(f"\n✅ Pipeline completed successfully")
            
            # Get the last step's output key for the final result
            last_output_key = None
            if steps:
                last_step = steps[-1]
                last_output_key = last_step.get('output_key', f"{last_step.get('name', 'final')}_output")
            
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
                first_output_key = steps[0].get('output_key', f"{steps[0].get('name', 'first')}_output")
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
    
    # Process step methods (the actual CoT pipeline implementation)
    def _process_example_step(self, client, step, previous_results, temperature, max_tokens):
        """Process a step that establishes a pattern with examples"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Get step parameters
            prompt_text = step.get('prompt', '')
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)
            
            if not prompt_text:
                return {
                    'success': False,
                    'error': 'Empty prompt in example step'
                }
            
            # Create system message with examples
            system_msg = SystemMessage(content=
                "You are an AI assistant that uses chain-of-thought reasoning to solve problems. "
                "Below are examples of the reasoning pattern to follow. "
                "For new problems, follow the same step-by-step approach."
            )
            
            # Create user message with examples, including context data if available
            user_content = "### EXAMPLES OF REASONING PATTERN:\n\n" + prompt_text
            if context_data:
                if isinstance(context_data, str):
                    context_str = context_data
                else:
                    context_str = json.dumps(context_data, indent=2)
                user_content += f"\n\n### CONTEXT DATA TO USE:\n\n{context_str}"
            
            user_msg = HumanMessage(content=user_content)
            
            # Add messages to history
            self.message_history.append(system_msg)
            self.message_history.append(user_msg)
            
            # Track raw history for debugging
            self.raw_history.append({
                'role': 'system', 
                'content': system_msg.content
            })
            self.raw_history.append({
                'role': 'user', 
                'content': user_msg.content
            })
            
            # Generate response using the client
            messages = [
                {"role": "system", "content": system_msg.content},
                {"role": "user", "content": user_msg.content}
            ]
            
            ai_response = client.generate_completion(
                messages=messages,
                temperature=step_temperature,
                max_tokens=step_max_tokens
            )
            
            # Add AI response to history
            from langchain_core.messages import AIMessage
            ai_msg = AIMessage(content=ai_response)
            self.message_history.append(ai_msg)
            
            # Track raw history for debugging
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
                'error': f"Error processing example step: {str(e)}"
            }
    
# This code should replace the existing _process_new_problem_step method in processors/cot_processor.py

    def _process_new_problem_step(self, client, step, previous_results, temperature, max_tokens):
        """Process a new problem step with improved schema handling"""
        from langchain_core.messages import HumanMessage, AIMessage
        try:
            self._prune_message_history(max_messages=5)
            prompt_text = step.get('prompt', '')
            input_key = step.get('input_key', '')
            schema = step.get('schema')
            step_temperature = step.get('temperature', temperature)
            step_max_tokens = step.get('max_tokens', max_tokens)
            context_data = step.get('context_data', None)
            
            if not prompt_text:
                return {'success': False, 'error': 'Empty prompt in new problem step'}
            
            if input_key and input_key in previous_results:
                # Extract just the conclusion rather than full previous result
                prev_result = previous_results[input_key]
                conclusion = self._extract_conclusion(prev_result)
                prompt_text = f"Previous result conclusion:\n```\n{conclusion}\n```\n\n{prompt_text}"
                
            if "previous analysis" not in prompt_text.lower():
                prompt_text = f"Continue the analysis based on the previous conclusion. Be concise.\n\n{prompt_text}"

            if context_data:
                context_str = context_data if isinstance(context_data, str) else json.dumps(context_data, indent=2)
                prompt_text = f"Context Data:\n```\n{context_str}\n```\n\n{prompt_text}"
            
            # Enhance prompt for JSON schema if present
            if schema:
                schema_str = json.dumps(schema, indent=2)
                prompt_text += f"\n\nYou MUST format your response as a valid JSON object that conforms to the following schema:\n```json\n{schema_str}\n```\n"
                prompt_text += "Important: Your response must be a properly formatted JSON object without any additional text, markdown, or explanation."
            
            user_msg = HumanMessage(content=prompt_text)
            self.message_history.append(user_msg)
            self.raw_history.append({'role': 'user', 'content': user_msg.content})
            
            # Convert the message history to a format the client can use
            messages = []
            for msg in self.message_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    messages.append({"role": "system", "content": msg.content})
            
            # Generate response using the client
            if schema:
                response = client.generate_completion(
                    messages=messages,
                    temperature=step_temperature,
                    max_tokens=step_max_tokens,
                    response_format={"type": "json_schema", "json_schema": schema}
                )
                structured_output = response
            else:
                response = client.generate_completion(
                    messages=messages,
                    temperature=step_temperature,
                    max_tokens=step_max_tokens
                )
                structured_output = response
            
            # Add AI response to history
            ai_msg = AIMessage(content=str(response) if isinstance(response, dict) else response)
            self.message_history.append(ai_msg)
            
            # Use the actual content for the raw history, not the stringified version
            self.raw_history.append({
                'role': 'assistant', 
                'content': response if isinstance(response, str) else json.dumps(response, indent=2)
            })
            
            return {'success': True, 'output': structured_output}
            
        except Exception as e:
            return {'success': False, 'error': f"Error processing new problem step: {str(e)}"}


    def _process_tonality_step(self, client, step, previous_results, temperature, max_tokens):
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
            
            # Add context data to the user message if available
            user_content = f"Convert the following drug information to a standardized list:\n{input_str}"
            if context_data:
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
            
            # Generate response using the client
            ai_response = client.generate_completion(
                messages=api_messages,
                temperature=step_temperature,
                max_tokens=step_max_tokens
            )
            
            # Add a simplified version to history to maintain continuity
            from langchain_core.messages import HumanMessage, AIMessage
            history_user_msg = HumanMessage(content="Please format the previous information according to our style guidelines.")
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
    
# This code should replace the existing _process_regular_step method in processors/cot_processor.py

    def _process_regular_step(self, client, step, previous_results, step_label="QUESTION", temperature=0, max_tokens=8000):
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
                logger.info(f"Using previous result from '{input_key}' for step")
                
                # Extract just the conclusion rather than full previous result
                conclusion = self._extract_conclusion(previous_result)
                
                # Incorporate previous result into the prompt
                modified_prompt = (
                    f"Based on previous conclusion:\n\n"
                    f"```\n{conclusion}\n```\n\n"
                    f"{prompt_text}"
                )

            # Add context data to the prompt if available
            if context_data:
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
            user_msg = HumanMessage(content=
                f"### {step_label}:\n\n" + modified_prompt
            )
            
            # Add message to history
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
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    messages.append({"role": "system", "content": msg.content})
            
            # Generate response using the client
            if schema:
                # Use schema for structured output
                ai_response = client.generate_completion(
                    messages=messages,
                    temperature=step_temperature,
                    max_tokens=step_max_tokens,
                    response_format={"type": "json_schema", "json_schema": schema}
                )
            else:
                # Standard text generation
                ai_response = client.generate_completion(
                    messages=messages,
                    temperature=step_temperature,
                    max_tokens=step_max_tokens
                )
            
            # Add AI response to history
            ai_msg = AIMessage(content=str(ai_response) if isinstance(ai_response, dict) else ai_response)
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

    def _process_verification_step(self, client, step, previous_results, temperature=0, max_tokens=8000):
        """Process a verification step"""
        return self._process_regular_step(client, step, previous_results, "VERIFICATION CHECK", temperature, max_tokens)
    
    def _process_summary_step(self, client, step, previous_results, temperature=0, max_tokens=8000):
        """Process a summary step"""
        return self._process_regular_step(client, step, previous_results, "SUMMARY", temperature, max_tokens)


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
        system_messages = [msg for msg in self.message_history if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in self.message_history if not isinstance(msg, SystemMessage)]
        
        # Keep the last max_messages pairs (human + AI)
        recent_messages = other_messages[-max_messages*2:] if other_messages else []
        
        # Reconstruct history
        self.message_history = system_messages + recent_messages
        
        # Also prune raw history
        if len(self.raw_history) > max_messages*2 + len(system_messages):
            system_entries = [entry for entry in self.raw_history if entry.get('role') == 'system']
            other_entries = [entry for entry in self.raw_history if entry.get('role') != 'system']
            recent_entries = other_entries[-max_messages*2:] if other_entries else []
            self.raw_history = system_entries + recent_entries