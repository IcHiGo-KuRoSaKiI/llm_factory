
# clients/azure_client.py
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import AzureOpenAI

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class AzureLLMClient(BaseLLMClient):
    """Azure OpenAI client implementation"""
    
    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_version: Optional[str] = None,
                 deployment_name: Optional[str] = None,
                 default_temperature: float = 0,
                 default_max_tokens: int = 8000):
        """Initialize Azure OpenAI client"""
        try:
            # Initialize the direct OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_BASE_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_API_KEY"),
                api_version=api_version or os.getenv("AZURE_API_VERSION")
            )
            self.deployment_name = deployment_name or os.getenv("AZURE_GPT_DEPLOYMENT_NAME")
            
            # Initialize the LangChain client
            self.llm_params = {
                "azure_endpoint": azure_endpoint or os.getenv("AZURE_BASE_ENDPOINT"),
                "api_key": api_key or os.getenv("AZURE_API_KEY"),
                "api_version": api_version or os.getenv("AZURE_API_VERSION"),
                "deployment_name": deployment_name or os.getenv("AZURE_GPT_DEPLOYMENT_NAME"),
                "temperature": default_temperature,
                "max_tokens": default_max_tokens
            }
            self.langchain_client = AzureChatOpenAI(**self.llm_params)
            
            # Settings
            self.default_temperature = default_temperature
            self.default_max_tokens = default_max_tokens
            
            # Initialize conversation history storage
            self.conversation_history = {}
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {str(e)}")
    
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
                            response_format: Optional[Dict[str, Any]] = None,
                            max_attempts: int = 3,
                            subsection_name: str = "Unknown",
                            conversation_id: str = "default",
                            **kwargs) -> Union[str, Dict]:
        """Generate a completion using Azure OpenAI"""
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
            langchain_messages = self.conversation_history[conversation_id] + langchain_messages
        
        while attempts < max_attempts:
            try:
                # Set up the LangChain client with the requested parameters
                llm = self.langchain_client.bind(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Handle JSON schema if provided
                if response_format and response_format.get("type") == "json_schema":
                    schema = response_format.get("json_schema", {})
                    llm_with_schema = llm.with_structured_output(
                        schema=schema,
                        include_raw=True
                    )
                    
                    response = llm_with_schema.invoke(langchain_messages)
                    
                    # Extract content from response
                    if isinstance(response, dict) and "raw" in response:
                        ai_msg_content = response["raw"].content if hasattr(response["raw"], "content") else str(response["raw"])
                        result = {k: v for k, v in response.items() if k != "raw"}
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
                        self.conversation_history[conversation_id].append(langchain_messages[0])
                    
                    # Add the latest user messages and AI response
                    for msg in langchain_messages[1:]:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            self.conversation_history[conversation_id].append(msg)
                else:
                    # Just append the new messages to existing history
                    for msg in langchain_messages:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            self.conversation_history[conversation_id].append(msg)
                
                # Add AI response to history
                self.conversation_history[conversation_id].append(ai_msg)
                
                return result
                
            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempts+1}): {str(e)}")
                attempts += 1
                time.sleep(1)
        
        # Return placeholder if all attempts failed
        error_msg = f"Failed to generate completion for '{subsection_name}' after {max_attempts} attempts."
        logger.error(error_msg)
        
        if response_format and response_format.get("type") == "json_schema":
            return {"error": error_msg, "status": "failed"}
        else:
            return f"[Content generation failed: {error_msg}]"
