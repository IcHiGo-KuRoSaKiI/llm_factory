import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Clean up existing handlers to avoid duplication
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(logging.Formatter(
            '%(levelname)s\t%(message)s'
        ))
        break

# Base classes for the Factory Method pattern
class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate_completion(self, messages, temperature=0, max_tokens=8000, **kwargs):
        """Generate a completion using the LLM"""
        pass

class PromptProcessor(ABC):
    """Abstract base class for prompt processors"""
    
    @abstractmethod
    def process(self, client: LLMClient, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a prompt configuration using the provided LLM client"""
        pass

# Factory for creating LLM clients
class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(client_type: str, **kwargs) -> LLMClient:
        """Create an LLM client of the specified type"""
        if client_type.lower() == "azure":
            from clients.azure_client import AzureLLMClient
            return AzureLLMClient(**kwargs)
        elif client_type.lower() == "groq":
            from clients.groq_client import GroqLLMClient
            return GroqLLMClient(**kwargs)
        elif client_type.lower() == "openai":
            from clients.openai_client import OpenAILLMClient
            return OpenAILLMClient(**kwargs)
        elif client_type.lower() in ["lmstudio", "local"]:
            from clients.lmstudio_client import LMStudioClient
            return LMStudioClient(**kwargs)
        elif client_type.lower() in ["ollama"]:
            from clients.ollama_client import OllamaLLMClient
            return OllamaLLMClient(**kwargs)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

# Factory for creating prompt processors
class PromptProcessorFactory:
    """Factory for creating prompt processors"""
    
    @staticmethod
    def create_processor(pipeline_type: str) -> PromptProcessor:
        """Create a prompt processor of the specified type"""
        if pipeline_type.lower() == "standard":
            from processors.standard_processor import StandardPromptProcessor
            return StandardPromptProcessor()
        elif pipeline_type.lower() in ["cot", "chain_of_thought", "multi_step"]:
            from processors.cot_processor import ChainOfThoughtProcessor
            return ChainOfThoughtProcessor()
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

# Main function to run a pipeline
def run_pipeline(
    prompt_config: Optional[Dict[str, Any]] = None,
    input_path_or_text: Optional[str] = None,
    client_type: str = "azure",
    pipeline_type: str = "standard",
    output_path: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 8000,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a prompt pipeline using the specified LLM client and processor.
    
    Args:
        prompt_config: The prompt configuration dictionary
        input_path_or_text: Path to input file or raw text input
        client_type: Type of LLM client to use (azure, groq, openai, lmstudio)
        pipeline_type: Type of prompt pipeline to use (standard, cot/multi_step)
        output_path: Path to save the output (optional)
        temperature: Temperature parameter for the LLM
        max_tokens: Maximum tokens parameter for the LLM
        **kwargs: Additional parameters to pass to the LLM client
        
    Returns:
        The results of the pipeline execution
    """
    try:
        # Load input content if a path is provided
        if input_path_or_text and os.path.isfile(input_path_or_text):
            with open(input_path_or_text, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = input_path_or_text
            
        # Load prompt configuration if a path is provided
        if isinstance(prompt_config, str) and os.path.isfile(prompt_config):
            with open(prompt_config, 'r', encoding='utf-8') as f:
                prompt_config = json.load(f)
        
        # If prompt_config is still None, assume we're using content directly
        if prompt_config is None and content:
            if pipeline_type.lower() == "standard":
                # For standard prompts, use content as the prompt
                prompt_config = {
                    "name": "simple_prompt",
                    "prompt": content,
                    "context_data": kwargs.get("context_data")
                }
            else:
                # Try to parse content as a pipeline configuration
                try:
                    prompt_config = json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("For multi-step pipelines, input must be a valid JSON pipeline configuration")
        
        # Create the client and processor
        logger.info(f"Creating {client_type} client")
        client = LLMClientFactory.create_client(
            client_type=client_type,
            **kwargs
        )
        
        logger.info(f"Creating {pipeline_type} processor")
        processor = PromptProcessorFactory.create_processor(pipeline_type)
        
        # Process the prompt
        logger.info(f"Processing prompt with {pipeline_type} processor")
        results = processor.process(
            client=client,
            prompt_config=prompt_config,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Save the results if an output path is provided
        if output_path:
            logger.info(f"Saving results to {output_path}")
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise