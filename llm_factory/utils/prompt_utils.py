# utils/prompt_utils.py
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import json

from llm_factory.core import LLMClientFactory
from llm_factory.env_loader import ENV_VARS

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """
    A utility class for enhancing prompts based on context data and user feedback.

    This class helps improve a base prompt by analyzing optional context data 
    and incorporating new guidelines or feedback from the user.
    """

    def __init__(self, client=None, client_type: str = None, dry_run: bool = False, **client_kwargs):
        """
        Initialize the PromptEnhancer with an optional LLM client.

        Args:
            client: An existing LLM client instance to use
            client_type: The type of client to create if none provided
            **client_kwargs: Additional kwargs to pass to the client creation
        """
        self.client = client
        self.client_type = client_type or ENV_VARS.get('default_client_type', 'azure')
        self.client_kwargs = client_kwargs
        self.dry_run = dry_run

        # Create client if not provided
        if self.client is None:
            try:
                self.client = LLMClientFactory.create_client(
                    client_type=client_type,
                    **client_kwargs
                )
                logger.info(
                    f"Created {client_type} client for prompt enhancement")
            except Exception as e:
                logger.error(f"Failed to create client: {str(e)}")
                raise RuntimeError(f"Failed to create LLM client: {str(e)}")

        # Try to detect client type if not provided
        if not hasattr(self, 'client_type') or not self.client_type:
            self.client_type = self._detect_client_type()

        logger.info(f"Using client type: {self.client_type}")

    def _detect_client_type(self) -> str:
        """Try to detect the client type from the client instance"""
        client_class = self.client.__class__.__name__.lower()

        if "azure" in client_class:
            return "azure"
        elif "openai" in client_class:
            return "openai"
        elif "lmstudio" in client_class:
            return "lmstudio"
        elif "groq" in client_class:
            return "groq"
        elif "ollama" in client_class:
            return "ollama"
        else:
            return "unknown"

    def _format_schema_for_client(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Format the schema for the specific client type"""
        client_type = getattr(self, 'client_type', 'unknown').lower()

        if client_type == "azure":
            # Azure requires a specific format with title and description
            if "title" not in schema:
                schema["title"] = "EnhancedPromptSchema"
            if "description" not in schema:
                schema["description"] = "Schema for enhanced prompt output"
            return schema
        elif client_type == "lmstudio":
            # LM Studio format wraps schema in another layer
            return {
                "name": "enhanced_prompt_schema",
                "schema": schema
            }
        else:
            # Default to the original schema
            return schema

    def enhance_prompt(
        self,
        base_prompt: str,
        new_prompt: str,
        context_data: Any = None,
        enhancement_type: str = "general",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance a base prompt using context data and new guidelines.

        Args:
            base_prompt: The original prompt to enhance
            new_prompt: New guidelines or feedback to incorporate
            context_data: Optional context data (document, code, etc.)
            enhancement_type: Type of enhancement (general, document, code, creative)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            explain: Whether to include explanations of changes

        Returns:
            Dict with enhanced prompt and explanations
        """
        try:
            # Check if dry-run mode is enabled
            if self.dry_run:
                logger.info("🔍 DRY-RUN MODE: Simulating prompt enhancement")
                return {
                    "enhanced_prompt": base_prompt,  # Return original prompt in dry-run
                    "explanation": "DRY-RUN: Prompt enhancement was simulated, no actual API call made",
                    "reasoning": f"In dry-run mode, returning original prompt unchanged. Would have enhanced with: {new_prompt[:100]}...",
                    "dry_run": True
                }
            
            # Validate inputs
            if not base_prompt:
                raise ValueError("Base prompt cannot be empty")
            if not new_prompt:
                raise ValueError("New prompt/guidelines cannot be empty")

            # Format context data if provided
            context_str = ""
            if context_data:
                if isinstance(context_data, str):
                    context_str = context_data
                else:
                    try:
                        context_str = json.dumps(context_data, indent=2)
                    except:
                        context_str = str(context_data)

            # Create meta prompt to guide enhancement
            meta_prompt = self._create_meta_prompt(
                enhancement_type=enhancement_type,
                explain=explain
            )

            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": meta_prompt},
                {"role": "user", "content": self._format_user_message(
                    base_prompt=base_prompt,
                    new_prompt=new_prompt,
                    context_str=context_str,
                    explain=explain
                )}
            ]

            # Define a schema for structured output
            response_schema = {
                "type": "object",
                "properties": {
                    "enhanced_prompt": {
                        "type": "string",
                        "description": "The enhanced prompt text"
                    }
                },
                "required": ["enhanced_prompt"]
            }

            # Add explanation field to schema if requested
            if explain:
                response_schema["properties"]["explanation"] = {
                    "type": "string",
                    "description": "Explanation of changes made to the prompt"
                }
                response_schema["properties"]["reasoning"] = {
                    "type": "string",
                    "description": "Reasoning behind the enhancements"
                }
                response_schema["required"].extend(
                    ["explanation", "reasoning"])

            # Format schema for specific client
            formatted_schema = self._format_schema_for_client(response_schema)

            # Generate enhanced prompt
            logger.info(
                f"Generating enhanced prompt of type: {enhancement_type} with client: {self.client_type}")
            try:
                response = self.client.generate_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_schema",
                                     "json_schema": formatted_schema}
                )
                logger.info(f"Received response of type: {type(response)}")
            except Exception as client_error:
                logger.error(
                    f"Client error during completion: {str(client_error)}")
                raise

            # Ensure we got a valid response
            if isinstance(response, dict):
                # Check if response has a 'parsed' field (occurs with some LLM clients)
                if "parsed" in response and isinstance(response["parsed"], dict):
                    response = response["parsed"]

                # Now check for enhanced_prompt in the response
                if "enhanced_prompt" not in response:
                    raise ValueError(
                        f"Response missing enhanced_prompt: {response}")

                logger.info("Successfully enhanced prompt")
                return response
            else:
                raise ValueError(
                    f"Invalid response format from LLM: {type(response)}")

        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            # In case of error, provide debugging information
            error_info = {
                "error": str(e),
                "enhanced_prompt": base_prompt,  # Return original as fallback
                "explanation": f"Failed to enhance prompt due to: {str(e)}"
            }

            # Try to add the raw response if available
            if 'response' in locals():
                error_info["raw_response"] = response

            return error_info

    def _create_meta_prompt(self, enhancement_type: str = "general", explain: bool = True) -> str:
        """Create the system meta-prompt based on enhancement type"""
        base_instructions = (
            "You are an expert prompt engineer specializing in improving and refining prompts "
            "for large language models. Your task is to analyze a base prompt, context data, "
            "and new guidelines to create an enhanced version of the base prompt."
        )

        type_specific_instructions = {
            "general": (
                "Focus on making the prompt more specific, effective, and aligned with "
                "the new guidelines while preserving the original intent."
            ),
            "document": (
                "Specialize in enhancing prompts for document analysis tasks. Focus on "
                "document structure, information extraction, and summarization capabilities. "
                "Ensure the enhanced prompt guides the LLM to properly process document format, "
                "sections, tables, and other structural elements."
            ),
            "code": (
                "Specialize in enhancing prompts for code-related tasks. Focus on improving "
                "code generation, analysis, debugging, and explanation capabilities. "
                "Ensure the enhanced prompt guides the LLM to produce high-quality, "
                "maintainable, and efficient code with proper documentation."
            ),
            "creative": (
                "Specialize in enhancing prompts for creative writing tasks. Focus on improving "
                "storytelling, character development, dialogue, and narrative structure. "
                "Ensure the enhanced prompt guides the LLM to produce engaging, coherent, "
                "and imaginative content."
            )
        }

        # Get type-specific instructions or use general if type not found
        type_instructions = type_specific_instructions.get(
            enhancement_type.lower(), type_specific_instructions["general"]
        )

        explanation_instructions = ""
        if explain:
            explanation_instructions = (
                "In addition to the enhanced prompt, provide:\n"
                "1. A detailed explanation of the specific changes you made\n"
                "2. Your reasoning for each major enhancement\n"
                "3. How the enhanced prompt better addresses the context and new guidelines"
            )

        return f"""
{base_instructions}

{type_instructions}

GUIDELINES FOR PROMPT ENHANCEMENT:
1. Maintain the original purpose and core functionality of the base prompt
2. Incorporate the new guidelines and feedback effectively
3. Consider the context data (if provided) to make the prompt more specific
4. Add structural elements like step-by-step instructions when beneficial
5. Include relevant examples if they would improve prompt effectiveness
6. Use clear, concise language and proper formatting
7. Remove any redundant or confusing elements from the original prompt
8. Ensure the enhanced prompt is complete and self-contained

{explanation_instructions}

Your response must be in valid JSON format matching the requested schema.
"""

    def _format_user_message(
        self,
        base_prompt: str,
        new_prompt: str,
        context_str: str = "",
        explain: bool = True
    ) -> str:
        """Format the user message with base prompt, new guidelines, and context"""
        message = f"""
## BASE PROMPT
```
{base_prompt}
```

## NEW GUIDELINES/FEEDBACK
```
{new_prompt}
```
"""

        if context_str:
            message += f"""
## CONTEXT DATA
```
{context_str}
```
"""

        message += """
Please enhance the base prompt by incorporating the new guidelines and considering the context data.
"""

        if explain:
            message += " Include explanations of your changes and your reasoning for the enhancements."

        return message


def enhance_prompt(
    base_prompt: str,
    new_prompt: str,
    context_data: Any = None,
    enhancement_type: str = "general",
    client=None,
    client_type: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    explain: bool = True,
    dry_run: bool = False,
    **client_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to enhance a prompt without explicitly creating a PromptEnhancer.

    Args:
        base_prompt: The original prompt to enhance
        new_prompt: New guidelines or feedback to incorporate
        context_data: Optional context data (document, code, etc.)
        enhancement_type: Type of enhancement (general, document, code, creative)
        client: An existing LLM client instance (optional)
        client_type: The type of client to create if none provided
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        explain: Whether to include explanations of changes
        **client_kwargs: Additional kwargs to pass to client creation

    Returns:
        Dict with enhanced prompt and explanations
    """
    enhancer = PromptEnhancer(
        client=client,
        client_type=client_type,
        dry_run=dry_run,
        **client_kwargs
    )

    return enhancer.enhance_prompt(
        base_prompt=base_prompt,
        new_prompt=new_prompt,
        context_data=context_data,
        enhancement_type=enhancement_type,
        temperature=temperature,
        max_tokens=max_tokens,
        explain=explain
    )
