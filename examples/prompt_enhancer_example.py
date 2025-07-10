# examples/prompt_enhancer_example.py
# python -m examples.prompt_enhancer_example
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Import from utils for the example
from llm_factory.utils import enhance_prompt, PromptEnhancer

# Load environment variables
load_dotenv(override=True)


def example_general_enhancement():
    """Example of enhancing a general prompt"""

    # Original basic prompt
    base_prompt = """
    Analyze the data and provide insights.
    """

    # New guidelines from user
    new_prompt = """
    I want the analysis to focus on trends over time and identify anomalies.
    Please also include visualizations and actionable recommendations.
    """

    # Sample data as context
    context_data = {
        "monthly_sales": [
            {"month": "Jan", "value": 10500},
            {"month": "Feb", "value": 11200},
            {"month": "Mar", "value": 10800},
            {"month": "Apr", "value": 12300},
            {"month": "May", "value": 14200},
            {"month": "Jun", "value": 13100}
        ]
    }

    print("\n=== Example 1: General Prompt Enhancement ===")
    print(f"Base prompt: {base_prompt}")
    print(f"New guidelines: {new_prompt}")

    # Enhance the prompt
    result = enhance_prompt(
        base_prompt=base_prompt,
        new_prompt=new_prompt,
        context_data=context_data,
        enhancement_type="general",
        temperature=0.3
    )

    # Display results
    print("\nEnhanced Prompt:")
    print(result.get("enhanced_prompt"))

    print("\nExplanation:")
    print(result.get("explanation"))


def example_document_analysis():
    """Example of enhancing a document analysis prompt"""

    # Original document analysis prompt
    base_prompt = """
    Summarize the main points of the document.
    """

    # New guidelines from user
    new_prompt = """
    I'm specifically interested in extracting financial metrics, risk factors, 
    and future growth projections. I need these organized by business segment.
    """

    # Sample document excerpt as context
    context_data = """
    ANNUAL REPORT 2024
    ACME CORPORATION
    
    FINANCIAL HIGHLIGHTS
    
    Revenue increased by 12% to $1.2 billion, with our Technology segment 
    contributing 55% of total revenue. EBITDA margin expanded to 28%, up from 
    25% in the previous fiscal year. The Board has approved a dividend of $1.20 
    per share, a 20% increase from last year.
    
    RISK FACTORS
    
    Market competition remains intense, particularly in our Consumer segment where 
    margins decreased by 2 percentage points. Supply chain disruptions continue to 
    pose challenges, although mitigation strategies reduced impact by 40% compared 
    to 2023.
    
    SEGMENT PERFORMANCE
    
    Technology Segment:
    - Revenue: $660M (↑15%)
    - Operating Margin: 32% (↑3%)
    - R&D Investment: $85M
    
    Consumer Segment:
    - Revenue: $420M (↑8%)
    - Operating Margin: 18% (↓2%)
    - New Product Launches: 12
    
    Industrial Segment:
    - Revenue: $120M (↑10%)
    - Operating Margin: 22% (↑1%)
    - Order Backlog: $45M (↑25%)
    """

    print("\n=== Example 2: Document Analysis Prompt Enhancement ===")

    # Create a PromptEnhancer instance (alternative approach)
    enhancer = PromptEnhancer(client_type="openrouter")

    # Enhance the prompt
    result = enhancer.enhance_prompt(
        base_prompt=base_prompt,
        new_prompt=new_prompt,
        context_data=context_data,
        enhancement_type="document",
        temperature=0.3
    )

    # Display results
    print("\nEnhanced Prompt:")
    print(result.get("enhanced_prompt"))

    print("\nReasoning:")
    print(result.get("reasoning"))


def example_code_enhancement():
    """Example of enhancing a code-related prompt"""

    # Original code prompt
    base_prompt = """
    Write Python code to process a CSV file.
    """

    # New guidelines from user
    new_prompt = """
    I need to handle large financial transaction CSVs with potential missing values.
    The code should calculate aggregate statistics grouped by transaction categories
    and detect potential fraudulent transactions based on unusual patterns.
    Please use pandas efficiently and include error handling.
    """

    # Sample code/data snippet as context
    context_data = """
    Sample CSV format:
    
    transaction_id,date,amount,category,merchant,is_flagged
    10001,2024-01-15,125.30,grocery,WholeMarket,0
    10002,2024-01-15,1420.99,electronics,TechWorld,0
    10003,2024-01-16,14.75,food,CoffeeBucks,0
    10004,2024-01-16,325.00,clothing,FashionStore,0
    10005,2024-01-16,4999.99,electronics,TechWorld,1
    """

    print("\n=== Example 3: Code-Related Prompt Enhancement ===")

    # Enhance the prompt
    result = enhance_prompt(
        base_prompt=base_prompt,
        new_prompt=new_prompt,
        context_data=context_data,
        enhancement_type="code",
        temperature=0.4,
        client_type="openrouter"
    )

    # Display results
    print("\nEnhanced Prompt:")
    print(result.get("enhanced_prompt"))


def save_enhanced_prompt(result: Dict[str, Any], output_path: str) -> None:
    """Save enhanced prompt results to a JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nEnhanced prompt saved to: {output_path}")
    except Exception as e:
        print(f"Error saving enhanced prompt: {str(e)}")


if __name__ == "__main__":
    # Run the examples
    try:
        # example_general_enhancement()
        example_document_analysis()
        # example_code_enhancement()

        # Save example result to file (optional)
        # save_enhanced_prompt(result, "enhanced_prompt.json")
    except Exception as e:
        print(f"Error running examples: {str(e)}")
