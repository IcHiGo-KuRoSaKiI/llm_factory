# Automatically load .env from the current working directory (usually the importing project's root)
import os
from dotenv import load_dotenv

# Only load .env if not already loaded (prevents overwriting existing env)
if not os.environ.get("LLM_FACTORY_ENV_LOADED"):
    load_dotenv()
    os.environ["LLM_FACTORY_ENV_LOADED"] = "1"

# Automatically load .env when llm_factory is imported
from .env_loader import load_environment
load_environment()

# Expose main API at package level
from .llm_factory import run_pipeline, LLMClientFactory, LLMClient, PromptProcessor, PromptProcessorFactory
from .parsers.pdf.parser import PDFParser
from .parsers.factory import ParserFactory