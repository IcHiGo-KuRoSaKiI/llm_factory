import os
from dotenv import load_dotenv

# This section attempts to load a .env file from the current working directory
# (which is typically the root of the project importing llm_factory).
# This allows the llm_factory to pick up the importing project's .env
# without the importing project needing to explicitly call load_dotenv()
# before importing llm_factory.
# It uses a flag to avoid doing this multiple times.
# `override=False` is important if you want to ensure that environment variables
# already set (e.g., by the system or a calling script) are not overwritten by .env values.
# However, python-dotenv's default behavior for load_dotenv() is to NOT override existing variables.
# If you want to ensure .env takes precedence for variables it defines, use override=True.
# For typical library use, not overriding (default or override=False) is safer.

if not os.environ.get("_LLM_FACTORY_INIT_DOTENV_LOADED"):
    # load_dotenv() will search for .env in the current working directory and its parents,
    # or a path specified by python_dotenv_path argument.
    # By default, it does not override existing environment variables.
    load_dotenv(verbose=False, override=False) # verbose=False to suppress "file not found" messages if .env doesn't exist
    os.environ["_LLM_FACTORY_INIT_DOTENV_LOADED"] = "1"

# Import and expose main API at package level.
# The env_loader.py module has `ENV_VARS = load_environment()` at its module level,
# so importing ENV_VARS or get_client_config will execute env_loader.py and
# populate ENV_VARS. The load_dotenv() call in env_loader.py will respect
# any variables already loaded by the section above or by the calling application.
from .env_loader import ENV_VARS, get_client_config

from .core import (
    run_pipeline,
    LLMClient, # This is the base class, kept for type hinting
    LLMClientFactory,
    PromptProcessor, # This is the base class, kept for type hinting
    PromptProcessorFactory
)

# Exposing specific client implementations and base class
from .clients import (
    BaseLLMClient, # Exposed as per README structure for extension
    AzureLLMClient,
    GroqLLMClient,
    LMStudioClient,
    OllamaLLMClient,
    OpenAILLMClient
)

# Exposing specific parsers and the parser factory
from .parsers.factory import ParserFactory
from .parsers.base_parser import BaseParser # Exposed for extension
from .parsers.pdf.parser import PDFParser
from .parsers.docx.parser import DocxParser
from .parsers.pptx.parser import PPTProcessor # Assuming PPTProcessor is the class for PPTX

# Exposing specific processor implementations and base class
from .processors import (
    BasePromptProcessor, # Exposed for extension
    StandardPromptProcessor,
    ChainOfThoughtProcessor
)

__all__ = [
    "run_pipeline",
    "LLMClientFactory",
    "LLMClient", # Base Class
    "BaseLLMClient", # Base Class for clients, for users to extend
    "AzureLLMClient",
    "GroqLLMClient",
    "LMStudioClient",
    "OllamaLLMClient",
    "OpenAILLMClient",
    "PromptProcessorFactory",
    "PromptProcessor", # Base Class
    "BasePromptProcessor", # Base Class for processors, for users to extend
    "StandardPromptProcessor",
    "ChainOfThoughtProcessor",
    "ParserFactory",
    "BaseParser", # Base Class for parsers, for users to extend
    "PDFParser",
    "DocxParser",
    "PPTProcessor",
    "ENV_VARS",
    "get_client_config"
]