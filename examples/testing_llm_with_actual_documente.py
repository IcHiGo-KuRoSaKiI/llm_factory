# python -m examples.testing_llm_with_actual_documente

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llm_factory import run_pipeline

# Load environment variables from .env file
load_dotenv(override=True)

# # Debug prints
# print(f"All environment variables: {dict(os.environ)}")
# print(f"DEFAULT_CLIENT_TYPE: {os.getenv('DEFAULT_CLIENT_TYPE')}")
# print(f"LM_STUDIO URL: {os.getenv('LM_STUDIO')}")



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

# Define Pydantic models for the graph structure
class Position(BaseModel):
    """Position with x and y coordinates"""
    x: float = Field(..., description="X-axis coordinate value")
    y: float = Field(..., description="Y-axis coordinate value")

class NodeChild(BaseModel):
    """Child node in graph structure"""
    id: str = Field(..., description="Unique identifier for the node")
    text: str = Field(..., description="Label or description of the node")
    position: Position = Field(..., description="Position of the node")

class GraphNode(BaseModel):
    """Graph node with position and children"""
    id: str = Field(..., description="Unique identifier for the node")
    text: str = Field(..., description="Label or description of the node")
    position: Position = Field(..., description="Position of the node")
    children: Optional[List[NodeChild]] = Field(default=[], description="Child nodes")

def format_schema_for_client(schema, client_type):
    """
    Format a schema based on the client type to ensure compatibility
    
    Args:
        schema: JSON schema (dict) or Pydantic model schema
        client_type: Type of client ("azure", "lmstudio", etc.)
        
    Returns:
        Properly formatted schema for the specified client
    """
    # If schema is a Pydantic model, get its JSON schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    
    if client_type.lower() == "azure":
        # Azure requires a specific format with title and description
        return {
            "title": schema.get("title", "OutputSchema"),
            "description": schema.get("description", "Schema for structured output"),
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    elif client_type.lower() == "lmstudio":
        # LM Studio just needs a regular JSON schema
        return schema
    else:
        # Default to the original schema
        return schema

# First prompt - analyze document
prompt1 = """You are an experienced project manager and solution architect. 
Analyze the provided document and identify key infrastructure components, their relationships, and requirements."""

# Second prompt - make more explicit instructions for JSON output
prompt2 = """Based on your analysis, create a graph structure representing the infrastructure components.

You MUST follow this exact format for each node:
{
  "id": "unique-string-id",
  "text": "Description of the node",
  "position": {
    "x": number,
    "y": number
  },
  "children": [
    // child nodes with the same structure
  ]
}

Start with a root node and add child nodes for each major component. Position nodes logically with x,y coordinates (use numbers between 0-1000).
"""

json_string = load_json_as_string('./azure_result.json')



# Set client type - change to "azure" when using Azure OpenAI
client_type = os.getenv("DEFAULT_CLIENT_TYPE", "lmstudio").lower()

# Get the Pydantic schema and format it for the client
pydantic_schema = GraphNode.model_json_schema()
formatted_schema = format_schema_for_client(pydantic_schema, client_type)

# For Azure, we need to adapt the schema to match their expected format
if client_type.lower() == "azure":
    # Azure needs a wrapper with name and schema for JSON schema
    azure_schema = {
        "name": "GraphNodeSchema",
        "schema": formatted_schema
    }
    formatted_schema = azure_schema

# Print schema for debugging
print(f"Formatted schema for {client_type}: {json.dumps(formatted_schema, indent=2)}")



# Create the CoT pipeline configuration

# print  (json_string)
cot_config = {
    "name": "problem_solver",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": prompt1,
            "output_key": "problem_analysis"
        },
         {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": "what did u understand from the document ?? ",
            "input_key": "problem_analysis",
            "output_key": "problem_analysis2",
        },
        {
            "type": "finalAnswer",
            "name": "final_answer",
            "prompt": prompt2,
            "input_key": "problem_analysis",
            "schema": formatted_schema,
            "output_key": "final_answer_output"
        },
        
    ],
    "context_data" : json_string
}

print(f"\n=== Example: Using {client_type.capitalize()} Model ===")

# Configure client parameters
if client_type.lower() == "lmstudio":
    client_config = {
        "base_url": os.environ.get("LM_STUDIO"),
        "temperature": 0,
        "max_tokens": 8000
    }
else:
    client_config = {
        "temperature": 0,
        "max_tokens": 8000
    }

# Run the pipeline
cot_result = run_pipeline(
    prompt_config=cot_config,
    client_type=client_type,
    pipeline_type="multi_step",
    **client_config
)

print ( cot_result )
final_json = cot_result["problem_solver"]["final_output"]
print("\nStructured Output:")
print(json.dumps(final_json, indent=2) if isinstance(final_json, dict) else final_json)