# clients/base_client.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    def generate_completion(self, 
                            messages: Union[List[Dict[str, str]], str, Dict[str, str]],
                            temperature: float = 0,
                            max_tokens: int = 8000,
                            **kwargs) -> Union[str, Dict]:
        """Generate a completion using the LLM"""
        pass

class BasePromptProcessor(ABC):
    """
    Abstract base class for prompt processors.
    Defines the interface that all prompt processors must implement.
    """
    
    @abstractmethod
    def process(self, client, prompt_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process a prompt configuration using the provided LLM client
        
        Args:
            client: The LLM client to use for text generation
            prompt_config: Configuration dictionary containing prompt details
            **kwargs: Additional parameters for the processing
            
        Returns:
            Dict[str, Any]: Results of processing the prompt
        """
        pass