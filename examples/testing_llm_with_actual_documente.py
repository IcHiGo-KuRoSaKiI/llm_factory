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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_as_string(file_path):
    """
    Load a JSON file from the given path and return it as a formatted string.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        str: JSON content as a formatted string
    """
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        # Convert the JSON data back to a formatted string
        json_string = json.dumps(json_data, indent=2)
        return json_string
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return None

# Example usage of the function
# json_string = load_json_as_string('/path/to/your/json/file.json')
# if json_string:
#     print(json_string)
#     # Use json_string as input for other operations

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


json_string = load_json_as_string('/Users/ichigo/Documents/GitHub/llm_factory/lmstudio_document_result.json')


prompt1 = """You are an experienced project manager and solution architect with a strong background in analyzing requirements and developing infrastructure solutions for various projects. 
Your task is to read and understand a Statement of Work (SOW) document and gather insights regarding the necessary infrastructure for project implementation. Here are the details you need to consider:  

SOW Document: __________  
Project Goals: __________  
Existing Infrastructure: __________  
Required Features: __________  
Budget Constraints: __________


The insights should be structured in a clear and concise format, including an overview of the project, identified requirements, recommended infrastructure components, potential challenges, and a timeline for implementation. 

Please ensure your analysis covers all critical aspects of the project and provides actionable recommendations based on the gathered insights. 

Constraints:  

Avoid technical jargon that may not be understood by all stakeholders.  
Keep the response focused on practical solutions rather than theoretical concepts.  
Be cautious of budget limitations and suggest cost-effective solutions.


Example insights you might include:  

"For the project to succeed, we recommend implementing a cloud-based infrastructure due to its scalability and flexibility."  
"The current infrastructure lacks the capacity to handle the projected workload; hence, we suggest upgrading the server capabilities."
"""

prompt2= """You are a knowledgeable data architect with extensive experience in creating structured data representations and graphs in JSON format. Your expertise lies in analyzing complex documents and extracting relevant infrastructure knowledge to represent it effectively.
Your task is to create a JSON graph based on the infrastructure knowledge extracted from the following document. Please review the document and follow the provided JSON schema to structure your output.

The JSON graph should clearly represent the relationships and entities discussed in the document. Use appropriate keys and values as specified in the JSON schema to ensure a coherent structure.
"""



graph_node_schema = {
  "name": "graph_node_schema",
  "schema": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string",
        "description": "Unique identifier for the node"
      },
      "text": {
        "type": "string",
        "description": "Label or description of the node"
      },
      "position": {
        "type": "object",
        "description": "2D coordinates specifying the position of the node in the graph",
        "properties": {
          "x": {
            "type": "number",
            "description": "X-axis coordinate"
          },
          "y": {
            "type": "number",
            "description": "Y-axis coordinate"
          }
        },
        "required": ["x", "y"]
      },
      "children": {
        "type": "array",
        "description": "List of child nodes nested under this node",
        "items": {
          "$ref": "#/schema" 
        }
      }
    },
    "required": ["id", "text", "position"]
  }
}


cot_config = {
    "name": "problem_solver",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": prompt1 ,
            "output_key": "problem_analysis",
            "context_data" : json_string,
        },
        {
            "type": "finalAnswer",
            "name": "final_answer",
            "prompt": prompt2 , 
            "input_key": "problem_analysis",  # Reference the previous step's output
            "schema": graph_node_schema,  # Pass the schema directly
            "output_key": "final_answer_output"
        }
    ]
}

print("\n=== Example: Using LM Studio Model ===")

lmstudio_config = {
    "base_url": os.environ.get("LM_SUDIO" ),
    "temperature": 0,
    "max_tokens": 8000
}

cot_result = run_pipeline(
    prompt_config=cot_config,
    client_type="lmstudio",
    pipeline_type="multi_step",
    **lmstudio_config
)


final_json = cot_result["problem_solver"]["final_output"]
print("\nStructured Output:")

print(final_json)
