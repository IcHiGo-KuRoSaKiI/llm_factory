import os
import time
import logging
from typing import Any, Dict
from dotenv import load_dotenv

from llm_factory.clients import OpenAILLMClient, LMStudioClient, OllamaLLMClient

logger = logging.getLogger(__name__)

class RoutingMixin:
    """Mixin providing environment-aware client routing for processors."""

    def __init__(self, *args, **kwargs):
        load_dotenv()
        self.env = self._load_env()
        self._init_clients()
        super().__init__(*args, **kwargs)

    def _load_env(self) -> Dict[str, Any]:
        return {
            "VISION_MODEL_ENABLED": os.getenv("VISION_MODEL_ENABLED", "false").lower() == "true",
            "VISION_CLIENT_TYPE": os.getenv("VISION_CLIENT_TYPE", "lmstudio").lower(),
            "VISION_MODEL_NAME": os.getenv("VISION_MODEL_NAME", "qwen-7b"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            "LM_STUDIO": os.getenv("LM_STUDIO", "http://localhost:1234"),
            "OLLAMA_HOST_URL": os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"),
            "OLLAMA_MODEL_NAME": os.getenv("OLLAMA_MODEL_NAME", "llama3.2"),
            "DEFAULT_TEMPERATURE": float(os.getenv("DEFAULT_TEMPERATURE", "0")),
            "DEFAULT_MAX_TOKENS": int(os.getenv("DEFAULT_MAX_TOKENS", "8000")),
            "API_REQUEST_TIMEOUT": int(os.getenv("API_REQUEST_TIMEOUT", "60")),
            "MAX_RETRIES": int(os.getenv("MAX_RETRIES", "3")),
            "RETRY_DELAY": int(os.getenv("RETRY_DELAY", "1")),
        }

    def _init_clients(self) -> None:
        self.openai_client = OpenAILLMClient(
            api_key=self.env["OPENAI_API_KEY"],
            model_name=self.env["OPENAI_MODEL_NAME"],
            default_temperature=self.env["DEFAULT_TEMPERATURE"],
            default_max_tokens=self.env["DEFAULT_MAX_TOKENS"],
        )

        self.lmstudio_client = LMStudioClient(
            base_url=self.env["LM_STUDIO"],
            model_name=self.env["VISION_MODEL_NAME"],
            default_temperature=self.env["DEFAULT_TEMPERATURE"],
            default_max_tokens=self.env["DEFAULT_MAX_TOKENS"],
        )

        self.ollama_client = OllamaLLMClient(
            host_url=self.env["OLLAMA_HOST_URL"],
            model_name=self.env["VISION_MODEL_NAME"],
            default_temperature=self.env["DEFAULT_TEMPERATURE"],
            default_max_tokens=self.env["DEFAULT_MAX_TOKENS"],
        )

    # ----- helper methods -----
    def get_completion(self, prompt: str, **kwargs) -> str:
        """Always route text completions to OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        params = {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.env["DEFAULT_TEMPERATURE"]),
            "max_tokens": kwargs.get("max_tokens", self.env["DEFAULT_MAX_TOKENS"]),
        }
        return self.openai_client.generate_completion(**params)

    def _vision_client(self):
        if self.env["VISION_CLIENT_TYPE"] == "lmstudio":
            return self.lmstudio_client
        if self.env["VISION_CLIENT_TYPE"] == "ollama":
            return self.ollama_client
        raise ValueError(f"Unsupported vision client type: {self.env['VISION_CLIENT_TYPE']}")

    def get_response_image(self, prompt: str, image_data: str, **kwargs) -> str:
        """Route image or PDF prompts to the configured vision backend."""
        client = self.openai_client
        if self.env["VISION_MODEL_ENABLED"]:
            client = self._vision_client()

        attempts = 0
        while attempts < self.env["MAX_RETRIES"]:
            try:
                return client.get_openai_response_image(
                    image_data=image_data,
                    prompt=prompt,
                    model=self.env["VISION_MODEL_NAME"],
                )
            except Exception as e:
                attempts += 1
                logger.warning(f"Vision request failed (attempt {attempts}): {e}")
                time.sleep(self.env["RETRY_DELAY"])

        logger.error("Vision client failed after retries")
        # Fallback to OpenAI if not already using it
        if client is not self.openai_client:
            return self.openai_client.get_openai_response_image(
                image_data=image_data,
                prompt=prompt,
                model=self.env["OPENAI_MODEL_NAME"],
            )
        raise RuntimeError("Image processing failed after retries")
