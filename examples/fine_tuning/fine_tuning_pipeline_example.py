# examples/fine_tuning/fine_tuning_pipeline_example.py
# python -m examples.fine_tuning.fine_tuning_pipeline_example
"""
Example demonstrating the new fine-tuning functionality for CoT pipelines.

This example shows how to use the 'fine_tune_prompt' keyword to automatically
enhance all prompts in a multi-step pipeline before execution.

Usage:
    python -m examples.fine_tuning_pipeline_example
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from llm_factory import run_pipeline

# Load environment variables
load_dotenv()

# Configure logging to see the fine-tuning process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_pipeline_without_fine_tuning() -> Dict[str, Any]:
    """Create a sample pipeline WITHOUT fine-tuning"""
    return {
        "name": "standard_analysis_pipeline",
        "steps": [
            {
                "type": "initialPrompt",
                "name": "problem_analysis",
                "prompt": "Analyze the given problem and break it down into components.",
                "output_key": "analysis_result"
            },
            {
                "type": "newQuestion",
                "name": "solution_development",
                "prompt": "Based on the analysis, develop a comprehensive solution.",
                "input_key": "analysis_result",
                "output_key": "solution_result"
            },
            {
                "type": "verification",
                "name": "solution_verification",
                "prompt": "Verify that the solution addresses all aspects of the original problem.",
                "input_key": "solution_result",
                "output_key": "verification_result"
            }
        ]
    }


def create_sample_pipeline_with_fine_tuning() -> Dict[str, Any]:
    """Create a sample pipeline WITH fine-tuning"""
    return {
        "name": "enhanced_analysis_pipeline",
        # NEW: This keyword enables automatic fine-tuning for all steps
        "fine_tune_prompt": """
        Focus on providing highly detailed, step-by-step technical analysis.
        Include specific examples, quantitative metrics where possible, and 
        clear reasoning chains. Ensure each response is comprehensive yet 
        well-structured for technical audiences.
        """,
        "steps": [
            {
                "type": "initialPrompt",
                "name": "problem_analysis",
                "prompt": "Analyze the given problem and break it down into components.",
                "output_key": "analysis_result"
            },
            {
                "type": "newQuestion",
                "name": "solution_development",
                "prompt": "Based on the analysis, develop a comprehensive solution.",
                "input_key": "analysis_result",
                "output_key": "solution_result"
            },
            {
                "type": "verification",
                "name": "solution_verification",
                "prompt": "Verify that the solution addresses all aspects of the original problem.",
                "input_key": "solution_result",
                "output_key": "verification_result"
            }
        ]
    }


def create_architecture_pipeline_with_fine_tuning() -> Dict[str, Any]:
    """Create a more complex architecture pipeline with fine-tuning"""

    # Sample schemas for structured output
    analysis_schema = {
        "name": "system_analysis_schema",
        "schema": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "responsibilities": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "complexity_level": {"type": "string"}
            }
        }
    }

    architecture_schema = {
        "name": "architecture_design_schema",
        "schema": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string"},
                "layers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "components": {"type": "array", "items": {"type": "string"}},
                            "technologies": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "data_flow": {"type": "string"}
            }
        }
    }

    return {
        "name": "software_architecture_pipeline",
        # Fine-tuning instruction for architecture-focused enhancement
        "fine_tune_prompt": """
        Transform all prompts to focus on enterprise software architecture best practices.
        Emphasize scalability, maintainability, security, and performance considerations.
        Include specific technology recommendations and architectural patterns.
        Ensure responses are detailed enough for senior engineers and architects.
        """,
        "steps": [
            {
                "type": "initialPrompt",
                "name": "requirements_analysis",
                "prompt": """
                Analyze the given requirements for a software system.
                Identify the key functional and non-functional requirements.
                Break down the system into logical components.
                """,
                "schema": analysis_schema,
                "output_key": "requirements_analysis"
            },
            {
                "type": "newQuestion",
                "name": "architecture_design",
                "prompt": """
                Design a comprehensive software architecture based on the requirements analysis.
                Define the system layers, components, and their interactions.
                Recommend appropriate technologies for each layer.
                """,
                "input_key": "requirements_analysis",
                "schema": architecture_schema,
                "output_key": "architecture_design"
            },
            {
                "type": "newQuestion",
                "name": "implementation_strategy",
                "prompt": """
                Create an implementation strategy for the proposed architecture.
                Define development phases, team structure, and technology adoption plan.
                Include risk mitigation strategies and success metrics.
                """,
                "input_key": "architecture_design",
                "output_key": "implementation_strategy"
            }
        ]
    }


def run_comparison_example():
    """Run both standard and fine-tuned pipelines for comparison"""

    # Sample problem to analyze
    problem_description = """
    We need to build a real-time chat application that can handle 100,000 concurrent users.
    The application should support text messages, file sharing, group chats, and voice calls.
    It needs to be scalable, secure, and work across web and mobile platforms.
    Users should be able to create channels, manage permissions, and integrate with external tools.
    """

    print("=" * 80)
    print("🔬 FINE-TUNING PIPELINE COMPARISON EXAMPLE")
    print("=" * 80)

    # Run standard pipeline (no fine-tuning)
    print("\n📋 Running STANDARD pipeline (no fine-tuning)...")
    print("-" * 50)

    standard_pipeline = create_sample_pipeline_without_fine_tuning()
    standard_pipeline["context_data"] = problem_description

    try:
        standard_result = run_pipeline(
            prompt_config=standard_pipeline,
            client_type="azure",  # Change to your preferred client
            pipeline_type="multi_step",
            temperature=0.3,
            max_tokens=2000
        )

        print("✅ Standard pipeline completed successfully")
        print(
            f"Final output preview: {str(standard_result.get('standard_analysis_pipeline', {}).get('final_output', 'No output'))[:200]}...")

    except Exception as e:
        print(f"❌ Standard pipeline failed: {str(e)}")
        standard_result = None

    # Run fine-tuned pipeline
    print("\n🔧 Running FINE-TUNED pipeline...")
    print("-" * 50)

    enhanced_pipeline = create_sample_pipeline_with_fine_tuning()
    enhanced_pipeline["context_data"] = problem_description

    try:
        enhanced_result = run_pipeline(
            prompt_config=enhanced_pipeline,
            client_type="azure",  # Change to your preferred client
            pipeline_type="multi_step",
            temperature=0.3,
            max_tokens=2000
        )

        print("✅ Fine-tuned pipeline completed successfully")
        pipeline_data = enhanced_result.get('enhanced_analysis_pipeline', {})
        print(
            f"Fine-tuning applied: {pipeline_data.get('fine_tuning_applied', False)}")
        print(f"Cache hits: {pipeline_data.get('cache_hits', 0)}")
        print(
            f"Final output preview: {str(pipeline_data.get('final_output', 'No output'))[:200]}...")

    except Exception as e:
        print(f"❌ Fine-tuned pipeline failed: {str(e)}")
        enhanced_result = None

    # Compare results if both succeeded
    if standard_result and enhanced_result:
        print("\n📊 COMPARISON SUMMARY")
        print("-" * 30)

        standard_final = standard_result.get(
            'standard_analysis_pipeline', {}).get('final_output', '')
        enhanced_final = enhanced_result.get(
            'enhanced_analysis_pipeline', {}).get('final_output', '')

        print(f"Standard output length: {len(str(standard_final))} characters")
        print(f"Enhanced output length: {len(str(enhanced_final))} characters")

        if len(str(enhanced_final)) > len(str(standard_final)):
            print("🎯 Fine-tuned pipeline produced more detailed output")

        # Save both results for detailed comparison
        with open("pipeline_comparison_results.json", "w") as f:
            json.dump({
                "standard_result": standard_result,
                "enhanced_result": enhanced_result,
                "comparison_summary": {
                    "standard_length": len(str(standard_final)),
                    "enhanced_length": len(str(enhanced_final)),
                    "enhancement_factor": len(str(enhanced_final)) / max(len(str(standard_final)), 1)
                }
            }, f, indent=2)

        print("\n💾 Detailed results saved to: pipeline_comparison_results.json")


def run_architecture_example():
    """Run the architecture pipeline example"""

    print("\n" + "=" * 80)
    print("🏗️ ARCHITECTURE PIPELINE WITH FINE-TUNING")
    print("=" * 80)

    # Sample requirements for architecture design
    requirements = """
    Requirements for an E-commerce Platform:
    
    Functional Requirements:
    - User registration and authentication
    - Product catalog with search and filtering
    - Shopping cart and checkout process
    - Payment processing with multiple gateways
    - Order management and tracking
    - Inventory management
    - Customer reviews and ratings
    - Admin dashboard for business management
    
    Non-Functional Requirements:
    - Handle 10,000 concurrent users
    - 99.9% uptime availability
    - Sub-second response times
    - PCI DSS compliance for payments
    - Support for web, mobile, and API clients
    - Scalable to multiple regions
    - Real-time inventory updates
    """

    architecture_pipeline = create_architecture_pipeline_with_fine_tuning()
    architecture_pipeline["context_data"] = requirements

    try:
        print("🚀 Running architecture pipeline with fine-tuning...")

        result = run_pipeline(
            prompt_config=architecture_pipeline,
            client_type="azure",  # Change to your preferred client
            pipeline_type="multi_step",
            temperature=0.2,  # Lower temperature for more consistent technical output
            max_tokens=3000
        )

        pipeline_data = result.get('software_architecture_pipeline', {})

        print("✅ Architecture pipeline completed successfully")
        print(
            f"🔧 Fine-tuning applied: {pipeline_data.get('fine_tuning_applied', False)}")
        print(f"📚 Cache entries: {pipeline_data.get('cache_hits', 0)}")

        # Display key results
        steps_data = pipeline_data.get('steps', {})

        if 'requirements_analysis' in steps_data:
            analysis = steps_data['requirements_analysis']
            if isinstance(analysis, dict) and 'components' in analysis:
                print(
                    f"\n📋 Components identified: {len(analysis.get('components', []))}")

        if 'architecture_design' in steps_data:
            design = steps_data['architecture_design']
            if isinstance(design, dict) and 'layers' in design:
                print(
                    f"🏗️ Architecture layers: {len(design.get('layers', []))}")

        # Save detailed results
        with open("architecture_pipeline_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(
            "\n💾 Detailed architecture results saved to: architecture_pipeline_result.json")

    except Exception as e:
        print(f"❌ Architecture pipeline failed: {str(e)}")
        logger.exception("Full error details:")


def demonstrate_fine_tuning_features():
    """Demonstrate specific fine-tuning features"""

    print("\n" + "=" * 80)
    print("🔍 FINE-TUNING FEATURES DEMONSTRATION")
    print("=" * 80)

    # Pipeline with multiple fine-tuning scenarios
    demo_pipeline = {
        "name": "fine_tuning_demo",
        "fine_tune_prompt": """
        Make all responses highly technical and include implementation details.
        Focus on providing code examples, specific technology choices, and 
        quantitative metrics. Structure responses with clear sections and bullet points.
        """,
        "steps": [
            {
                "type": "initialPrompt",
                "name": "technical_analysis",
                "prompt": "Analyze this technical problem and suggest approaches.",
                "output_key": "analysis"
            },
            {
                "type": "newQuestion",
                "name": "implementation_details",
                "prompt": "Provide implementation details for the suggested approach.",
                "input_key": "analysis",
                "output_key": "implementation"
            },
            {
                "type": "summary",
                "name": "final_recommendations",
                "prompt": "Summarize the recommendations with clear action items.",
                "input_key": "implementation",
                "output_key": "recommendations"
            }
        ],
        "context_data": """
        Problem: We need to optimize a REST API that's currently handling 1000 requests/second
        but frequently times out under load. The API serves a React frontend and mobile apps.
        Current stack: Node.js, Express, MongoDB, deployed on AWS EC2.
        """
    }

    try:
        print("🚀 Running fine-tuning demonstration...")

        result = run_pipeline(
            prompt_config=demo_pipeline,
            client_type="azure",
            pipeline_type="multi_step",
            temperature=0.4,
            max_tokens=2500
        )

        pipeline_data = result.get('fine_tuning_demo', {})

        print("✅ Demonstration completed successfully")

        # Show fine-tuning metadata
        if pipeline_data.get('fine_tuning_applied'):
            print("🔧 Fine-tuning Features:")
            print(f"   ✓ Applied to all {len(demo_pipeline['steps'])} steps")
            print(
                f"   ✓ Cache entries created: {pipeline_data.get('cache_hits', 0)}")
            print(
                f"   ✓ Instruction: {pipeline_data.get('fine_tune_instruction', '')[:50]}...")

        # Show step-by-step results
        steps_data = pipeline_data.get('steps', {})
        for step_key in ['analysis', 'implementation', 'recommendations']:
            if step_key in steps_data:
                output = str(steps_data[step_key])
                print(
                    f"\n📝 {step_key.title()} ({len(output)} chars): {output[:100]}...")

        # Save demonstration results
        with open("fine_tuning_demo_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print("\n💾 Demonstration results saved to: fine_tuning_demo_result.json")

    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    print("🚀 Starting Fine-Tuning Pipeline Examples")
    print("Make sure your .env file is configured with valid API credentials")

    try:
        # Run comparison between standard and fine-tuned pipelines
        run_comparison_example()

        # # Run architecture-specific example
        # run_architecture_example()

        # # Demonstrate fine-tuning features
        # demonstrate_fine_tuning_features()

        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("Check the generated JSON files for detailed results")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Example execution failed: {str(e)}")
        logger.exception("Full error details:")
