# python -m examples.actual_structures

import json
import logging
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from llm_factory import run_pipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the correct schema format (matching your working example)
json_schema = {
    "type": "object",
    "properties": {
        "component_name": {"type": "string"},
        "description": {"type": "string"},
        "tech_stack": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["component_name", "description", "tech_stack", "dependencies"]
}

# This is the proper CoT configuration with the correct schema format
cot_config = {
    "name": "problem_solver",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": "You are a helpful assistant who provides information about microservices architecture. You always structure your responses in a clear format.",
            "output_key": "problem_analysis"
        },
        {
            "type": "finalAnswer",
            "name": "final_answer",
            "prompt": "Describe a microservice for user authentication. Format your response as a plain JSON object with component_name, description, tech_stack (array), and dependencies (array). Do not include markdown code blocks or any other formatting.",
            "input_key": "problem_analysis",  # Reference the previous step's output
            "schema": json_schema,  # Pass the schema directly
            "output_key": "final_answer_output"
        }
    ]
}

print("\n=== Example: Using LM Studio Model ===")

lmstudio_config = {
    "base_url": os.environ.get("LM_SUDIO"),
    "temperature": 0,
    "max_tokens": 8000
}

cot_result = run_pipeline(
    prompt_config=cot_config,
    client_type="openrouter",
    pipeline_type="multi_step",
    # **lmstudio_config
)


final_json = cot_result["problem_solver"]["final_output"]
print("\nStructured Output:")

print(final_json)
