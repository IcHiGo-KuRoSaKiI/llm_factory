# env_loader.py
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment(env_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment variables from .env file and return them as a dictionary
    
    Args:
        env_path: Optional path to .env file (defaults to .env in current directory)
        
    Returns:
        Dictionary with environment variables
    """
    # Load environment variables from .env file
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        load_dotenv()
        logger.info("Loaded environment from default .env file")
    
    # Create a dictionary with all the environment variables
    env_vars = {
        # Azure OpenAI Configuration
        "azure": {
            "azure_endpoint": os.getenv("AZURE_BASE_ENDPOINT"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "deployment_name": os.getenv("AZURE_GPT_DEPLOYMENT_NAME"),
        },
        
        # Groq Configuration
        "groq": {
            "api_key": os.getenv("GROQ_API_KEY"),
            "model_name": os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192"),
        },
        
        # OpenAI Configuration
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        },
        
        # OpenRouter Configuration
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": os.getenv("OPENROUTER_MODEL_NAME", "qwen/qwen3-235b-a22b"),
            "site_url": os.getenv("OPENROUTER_SITE_URL", "https://github.com/IcHiGo-KuRoSaKiI/llm_factory"),
            "site_name": os.getenv("OPENROUTER_SITE_NAME", "LLM Factory"),
            "vision_model": os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini"),
        },
        
        # General LLM Settings
        "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0")),
        "default_max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "8000")),
        
        # Timeout Settings
        "api_request_timeout": int(os.getenv("API_REQUEST_TIMEOUT", "60")),
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "retry_delay": int(os.getenv("RETRY_DELAY", "1")),
        
        # Default Pipeline Configuration
        "default_pipeline_type": os.getenv("DEFAULT_PIPELINE_TYPE", "standard"),
        "default_client_type": os.getenv("DEFAULT_CLIENT_TYPE", "openrouter"),
    }
    
    # Check for missing required environment variables
    required_vars = []
    client_type = os.getenv("DEFAULT_CLIENT_TYPE", "openrouter").lower()
    
    if client_type == "azure":
        required_vars = ["AZURE_BASE_ENDPOINT", "AZURE_API_KEY", "AZURE_API_VERSION"]
    elif client_type == "groq":
        required_vars = ["GROQ_API_KEY"]
    elif client_type == "openai":
        required_vars = ["OPENAI_API_KEY"]
    elif client_type == "openrouter":
        required_vars = ["OPENROUTER_API_KEY"]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env_vars

def get_client_config(client_type: str, env_vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for a specific client type
    
    Args:
        client_type: The client type (azure, groq, openai, openrouter)
        env_vars: Dictionary with environment variables
        
    Returns:
        Dictionary with client configuration
    """
    client_type = client_type.lower()
    
    if client_type == "azure":
        return env_vars["azure"]
    elif client_type == "groq":
        return env_vars["groq"]
    elif client_type == "openai":
        return env_vars["openai"]
    elif client_type == "openrouter":
        return env_vars["openrouter"]
    else:
        raise ValueError(f"Unsupported client type: {client_type}")

# Load environment variables when this module is imported
ENV_VARS = load_environment()