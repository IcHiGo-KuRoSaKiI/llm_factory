# python -m examples.uml_output
from llm_factory import run_pipeline
import os
import json
import logging
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the factory pattern implementation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_standard_extraction(input_text, client_type="azure", output_path=None, temperature=0, max_tokens=8000, schema=None):
    """
    Run a standard extraction pipeline

    Args:
        input_text: The text to process
        client_type: The LLM client to use (azure, groq, openai)
        output_path: Optional path to save results
        temperature: Temperature setting for the LLM
        max_tokens: Maximum tokens for the LLM response
        schema: Optional JSON schema for structured output

    Returns:
        The extraction results
    """
    # Create a simple prompt configuration
    prompt_config = {
        "name": "simple_extraction",
        "prompt": "You are an AI assistant tasked with extracting and summarizing the main points from the provided content.",
        "context_data": input_text,
        "schema": schema
    }

    # Run the pipeline - remove temperature from kwargs since it's already a named parameter
    return run_pipeline(
        prompt_config=prompt_config,
        client_type=client_type,
        pipeline_type="standard",
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens
    )


def run_cot_pipeline(pipeline_config, client_type="azure", output_path=None, temperature=0, max_tokens=8000):
    """
    Run a Chain of Thought pipeline

    Args:
        pipeline_config: The CoT pipeline configuration
        client_type: The LLM client to use (azure, groq, openai)
        output_path: Optional path to save results
        temperature: Temperature setting for the LLM
        max_tokens: Maximum tokens for the LLM response

    Returns:
        The pipeline results
    """
    # Run the pipeline
    return run_pipeline(
        prompt_config=pipeline_config,
        client_type=client_type,
        pipeline_type="multi_step",
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens
    )


def run_local_model(text_input, base_url=None, temperature=0.7, max_tokens=2000):
    """
    Run a query against a local model using LM Studio

    Args:
        text_input: The text to process
        base_url: The URL of the local LM Studio server
        temperature: Temperature setting for sampling
        max_tokens: Maximum tokens for the response

    Returns:
        The model response
    """
    if base_url is None:
        base_url = os.environ.get("LM_SUDIO", "http://localhost:1234")

    return run_standard_extraction(
        input_text=text_input,
        client_type="lmstudio",
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )


# Example usage
def example_usage():
    """Examples of using the factory pattern"""

    # Example 1: Standard extraction without schema
    print("\n=== Example 1: Standard Extraction (No Schema) ===")
    standard_result = run_standard_extraction(
        input_text="Can u give me some sampels for the the Diagrams. They should be complex. Can u do that .",
        client_type="azure",
        temperature=0,
        schema=json_schema
    )
    # print("Standard Extraction Result (No Schema):",
    #       json.dumps(standard_result, indent=2))

    print(json.dumps(
        standard_result["simple_extraction"]["json_result"]["parsed"], indent=2))

    # # Example 2: Standard extraction with schema
    # print("\n=== Example 2: Standard Extraction (With Schema) ===")

    # # Define a simple schema for medication extraction
    # medication_schema = {
    #     "type": "object",
    #     "properties": {
    #         "medications": {
    #             "type": "array",
    #             "items": {
    #                 "type": "object",
    #                 "properties": {
    #                     "name": {"type": "string"},
    #                     "dosage": {"type": "string"},
    #                     "frequency": {"type": "string"}
    #                 },
    #                 "required": ["name", "dosage", "frequency"]
    #             }
    #         }
    #     },
    #     "required": ["medications"]
    # }

    # schema_result = run_standard_extraction(
    #     input_text="Patient takes Lisinopril 10mg once daily and Metformin 500mg twice daily.",
    #     client_type="azure",
    #     temperature=0,
    #     schema=medication_schema
    # )
    # print("Standard Extraction Result (With Schema):",
    #       json.dumps(schema_result, indent=2))

    # # Example 3: Chain of Thought pipeline
    # print("\n=== Example 3: Chain of Thought Pipeline ===")

    # cot_config = {
    #     "name": "problem_solver",
    #     "steps": [
    #         {
    #             "type": "initialPrompt",
    #             "name": "understand_problem",
    #             "prompt": "Analyze this math problem step by step: If x + y = 10 and x * y = 21, what are x and y?",
    #             "output_key": "problem_analysis"
    #         },
    #         {
    #             "type": "newQuestion",
    #             "name": "solve_problem",
    #             "prompt": "Now solve the problem using algebraic methods.",
    #             "input_key": "problem_analysis",
    #             "output_key": "solution"
    #         }
    #     ]
    # }

    # cot_result = run_cot_pipeline(
    #     pipeline_config=cot_config,
    #     client_type="azure",
    #     temperature=0
    # )

    # print("Chain of Thought Pipeline Result:",
    #       json.dumps(cot_result, indent=2))


if __name__ == "__main__":
    # Run the examples
    example_usage()
