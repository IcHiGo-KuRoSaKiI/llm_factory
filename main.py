# main.py
import os
import json
import logging
from typing import Any, Dict, Optional

# Import the factory pattern implementation
from llm_factory import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_standard_extraction(input_text, client_type="azure", output_path=None, temperature=0, max_tokens=8000):
    """
    Run a standard extraction pipeline
    
    Args:
        input_text: The text to process
        client_type: The LLM client to use (azure, groq, openai)
        output_path: Optional path to save results
        temperature: Temperature setting for the LLM
        max_tokens: Maximum tokens for the LLM response
        
    Returns:
        The extraction results
    """
    # Create a simple prompt configuration
    prompt_config = {
        "name": "simple_extraction",
        "prompt": "You are an AI assistant tasked with extracting and summarizing the main points from the provided content.",
        "context_data": input_text
    }
    
    # Run the pipeline
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


def run_local_model(text_input, base_url="http://localhost:1234", temperature=0.7, max_tokens=2000):
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
    return run_standard_extraction(
        input_text=text_input,
        client_type="lmstudio",
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )


# Example usage
def example_usage():
    # """Examples of using the factory pattern"""
    
    # # Example 1: Standard extraction
    # print("\n=== Example 1: Standard Extraction ===")
    # standard_result = run_standard_extraction(
    #     input_text="Patient takes Lisinopril 10mg once daily and Metformin 500mg twice daily.",
    #     client_type="azure",
    #     temperature=0
    # )
    # print("Standard Extraction Result:", json.dumps(standard_result, indent=2))
    
    # Example 2: Chain of Thought pipeline
    print("\n=== Example 2: Chain of Thought Pipeline ===")

    cot_config = {
        "name": "problem_solver",
        "steps": [
            {
                "type": "initialPrompt",
                "name": "understand_problem",
                "prompt": "Analyze this math problem step by step: If x + y = 10 and x * y = 21, what are x and y?",
                "output_key": "problem_analysis"
            },
            {
                "type": "newQuestion",
                "name": "solve_problem",
                "prompt": "Now solve the problem using algebraic methods.",
                "input_key": "problem_analysis",
                "output_key": "solution"
            },
            {
                "type": "verification",
                "name": "verify_solution",
                "prompt": "Verify that your solution is correct by substituting the values back into the original equations.",
                "input_key": "solution",
                "output_key": "verified_solution"
            },
            {
                "type": "finalAnswer",
                "name": "final_answer",
                "prompt": "Give me the final answwe, which gives me the values for X and Y}.",
                "input_key": "verified_solution"
            }
        ]
    }

    
    # cot_result = run_cot_pipeline(
    #     pipeline_config=cot_config,
    #     client_type="groq",
    #     temperature=0
    # )

    # print("Chain of Thought Pipeline Result:", json.dumps(cot_result, indent=2))
    

    print("\n=== Example : Using Ollama Studio Model ===")
    
    lmstudio_config = {
        # "base_url": "http://localhost:1234",  # Update this to your LM Studio server URL
        # "temperature": 0,
        # "max_tokens": 8000
    }

    cot_result = run_pipeline(
        prompt_config=cot_config,
        # client_type="lmstudio",
        client_type="ollama",
        pipeline_type="multi_step",
        **lmstudio_config
    )


    # cot_result = run_cot_pipeline(
    #         pipeline_config=cot_config,
    #         client_type="groq",
    #         temperature=0
    #     )
        
    print("CoT Result:", json.dumps(cot_result, indent=2))


    # # Example 3: Using a different client
    # print("\n=== Example 3: Using Groq Client ===")
    # groq_result = run_standard_extraction(
    #     input_text="Analyze the key features of this product: Ultra-slim laptop with 16GB RAM, 512GB SSD, Intel i7 processor, and 14-inch 4K display.",
    #     client_type="groq",
    #     temperature=0.3
    # )
    # print("Groq Result:", json.dumps(groq_result, indent=2))

if __name__ == "__main__":
    # Run the examples
    example_usage()