

## The structured output that i cannot confirm, works or not
import json
import logging
import os
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
from llm_factory import run_pipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging to see what's happening with the schema
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Coordinates(BaseModel):
    x: int
    y: int

class OutputWrapper(BaseModel):
    output: Coordinates

# Convert Pydantic schema to JSON schema
pydantic_schema = OutputWrapper.model_json_schema()
print(f"Raw Pydantic schema: {json.dumps(pydantic_schema, indent=2)}")

# Transform to LM Studio compatible format
lm_studio_schema = {
    "name": "coordinates_output",
    "strict": True,
    "schema": pydantic_schema
}

lm_studio_schema = {
    "name": "position_schema",
    "schema": {
        "type": "object",
        "properties": {
            "output": {
                "type": "object",
                "description": "Object representing a 2D coordinate point",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X-axis coordinate value"
                    },
                    "y": {
                        "type": "number",
                        "description": "Y-axis coordinate value"
                    }
                },
                "required": ["x", "y"]
            }
        },
        "required": ["output"]
    }
}


# print(f"LM Studio compatible schema: {json.dumps(json_schema, indent=2)}")

cot_config = {
    "name": "problem_solver",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": "You are a helpful assistant?",
            "output_key": "problem_analysis"
        },
        {
            "type": "finalAnswer",
            "name": "final_answer",
            "prompt": "Describe a microservice for user authentication",
            "input_key": "verified_solution",
            "schema": lm_studio_schema  # Use the transformed schema
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
    client_type="lmstudio",
    pipeline_type="multi_step",
    **lmstudio_config
)

print("CoT Result:", json.dumps(cot_result, indent=2))




# Make the result more accessible
def extract_final_output(result):
    """Extract the final output from a chain of thought pipeline result"""
    if not result:
        return None
        
    # If it's a dictionary with a single key (pipeline name)
    if isinstance(result, dict) and len(result) == 1:
        pipeline_result = next(iter(result.values()))
        if isinstance(pipeline_result, dict) and "final_output" in pipeline_result:
            return pipeline_result["final_output"]
    
    return result  # Return original if structure doesn't match expectations

# Get the final output directly
final_output = extract_final_output(cot_result)
print("\nFinal Output:", final_output)

