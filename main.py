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
        "name": "adult_dosing_indications_prompt",
        "steps": [
            {
                "type": "initialPrompt",
                "name": "extract_fda_indications",
                "prompt": "Extract all FDA-approved indications from the drug information.",
                "output_key": "fda_indications"
            },
            {
                "type": "newQuestion",
                "name": "add_off_label_uses",
                "prompt": "Now, identify any commonly used off-label uses for this medication.",
                "input_key": "fda_indications",
                "output_key": "indications_with_off_label"
            },
            {
                "type": "tonality",
                "name": "format_indications",
                "prompt": "Format the indications in a standardized list format suitable for healthcare professionals.",
                "input_key": "indications_with_off_label",
                "output_key": "formatted_indications"
            }
        ],
        "context_data": "Amiodarone is a class III antiarrhythmic medication used to treat and prevent several types of cardiac arrhythmias. The FDA has approved it for the treatment of ventricular arrhythmias, particularly ventricular tachycardia and ventricular fibrillation. It is also often used off-label for atrial fibrillation and atrial flutter."
    }
    
    cot_result = run_cot_pipeline(
        pipeline_config=cot_config,
        client_type="groq",
        temperature=0
    )
    print("Chain of Thought Pipeline Result:", json.dumps(cot_result, indent=2))
    
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