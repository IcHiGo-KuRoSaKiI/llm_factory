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
                            top_p: Optional[float] = None,
                            top_k: Optional[int] = None,
                            frequency_penalty: Optional[float] = None,
                            presence_penalty: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            min_p: Optional[float] = None,
                            top_a: Optional[float] = None,
                            seed: Optional[int] = None,
                            **kwargs) -> Union[str, Dict]:
        """Generate a completion using the LLM"""
        pass
    
    @abstractmethod
    def get_openai_response_image(self, 
                                 image_data: str, 
                                 prompt: Optional[str] = None,
                                 model: Optional[str] = None) -> str:
        """
        Extract text from an image using AI vision capabilities
        
        Args:
            image_data: Base64-encoded image data or data URI
            prompt: Optional custom prompt to use for image analysis
            model: Optional model name to use for image analysis
            
        Returns:
            Extracted text from the image
        """
        pass
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing unnecessary whitespace and formatting
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove redundant spaces
        cleaned = " ".join(text.split())
        
        # Remove common extraction artifacts
        cleaned = cleaned.replace("```", "").strip()
        
        return cleaned

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