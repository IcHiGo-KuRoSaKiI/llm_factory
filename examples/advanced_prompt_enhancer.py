# examples/advanced_prompt_enhancer.py
import os
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from llm_factory.utils import enhance_prompt, PromptEnhancer
from llm_factory import ParserFactory, LLMClientFactory, run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentPromptOptimizer:
    """
    A class that optimizes prompts for specific documents by:
    1. Parsing the document to extract content
    2. Analyzing the document structure and content
    3. Enhancing a base prompt to better suit the document
    """
    
    def __init__(self, client_type: str = "azure", **client_kwargs):
        """
        Initialize the optimizer with an LLM client.
        
        Args:
            client_type: Type of LLM client to use
            **client_kwargs: Additional kwargs for client creation
        """
        # Create LLM client for document processing
        self.client = LLMClientFactory.create_client(
            client_type=client_type,
            **client_kwargs
        )
        
        # Create a prompt enhancer using the same client
        self.enhancer = PromptEnhancer(client=self.client)
        
        # For storing document analysis
        self.document_analysis = {}
    
    def optimize_for_document(
        self,
        file_path: str,
        base_prompt: str,
        new_guidelines: str = "",
        enhancement_type: str = "document",
        analyze_document: bool = True,
        max_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize a prompt for a specific document.
        
        Args:
            file_path: Path to the document file
            base_prompt: Base prompt to enhance
            new_guidelines: User guidelines or feedback
            enhancement_type: Type of enhancement
            analyze_document: Whether to analyze document before enhancing
            max_chunks: Maximum number of document chunks to use as context
            
        Returns:
            Dictionary with optimized prompt and other information
        """
        try:
            # 1. Parse the document
            logger.info(f"Parsing document: {file_path}")
            document_content = self._parse_document(file_path)
            
            if not document_content:
                logger.warning("No content extracted from document")
                return {
                    "success": False,
                    "error": "Failed to extract content from document",
                    "enhanced_prompt": base_prompt
                }
            
            # 2. Analyze the document if requested
            document_analysis = None
            if analyze_document:
                logger.info("Analyzing document structure and content")
                document_analysis = self._analyze_document(document_content)
                self.document_analysis = document_analysis
            
            # 3. Prepare context for prompt enhancement
            context_data = self._prepare_context(document_content, document_analysis, max_chunks)
            
            # 4. Enhance the prompt
            logger.info("Enhancing prompt based on document analysis")
            enhanced_result = self.enhancer.enhance_prompt(
                base_prompt=base_prompt,
                new_prompt=new_guidelines or "Optimize this prompt for the provided document",
                context_data=context_data,
                enhancement_type=enhancement_type,
                temperature=0.4
            )
            
            # 5. Add additional information to the result
            result = {
                "success": True,
                "document_path": file_path,
                "document_analysis": document_analysis if analyze_document else "Not performed",
                **enhanced_result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing prompt for document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_prompt": base_prompt
            }
    
    def _parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a document to extract content"""
        try:
            # Create a parser for the document type
            parser = ParserFactory.create_parser(
                file_path=file_path, 
                openai_helper=self.client  # Use the LLM client for image processing
            )
            
            # Parse the document
            return parser.parse(file_path)
            
        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            return []
    
    def _analyze_document(self, document_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze document structure and content"""
        try:
            # Extract all text from document chunks
            all_text = "\n\n".join([chunk.get("text", "") for chunk in document_content])
            
            # Create a system prompt for document analysis
            analysis_prompt = {
                "name": "document_analyzer",
                "prompt": """
                You are an expert document analyzer. Examine the provided document content and create a structured analysis including:
                1. Document type and overall structure
                2. Key sections or components
                3. Content summary
                4. Special features (tables, charts, images, etc.)
                5. Writing style and tone
                6. Technical complexity level
                7. Target audience
                
                Provide your analysis in a well-structured JSON format.
                """,
                "context_data": all_text[:20000],  # Limit to avoid token issues
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_type": {"type": "string"},
                        "structure": {"type": "array", "items": {"type": "string"}},
                        "key_sections": {"type": "array", "items": {"type": "string"}},
                        "content_summary": {"type": "string"},
                        "special_features": {"type": "array", "items": {"type": "string"}},
                        "style_and_tone": {"type": "string"},
                        "complexity_level": {"type": "string"},
                        "target_audience": {"type": "string"},
                        "recommended_approach": {"type": "string"}
                    }
                }
            }
            
            # Run analysis pipeline
            analysis_result = run_pipeline(
                prompt_config=analysis_prompt,
                client=self.client,  # Use existing client
                pipeline_type="standard"
            )
            
            # Extract the analysis from results
            document_analyzer_key = "document_analyzer"
            if document_analyzer_key in analysis_result:
                result_key = "json_result" if "json_result" in analysis_result[document_analyzer_key] else "result"
                return analysis_result[document_analyzer_key][result_key]
            
            return {"error": "Analysis failed", "raw_result": analysis_result}
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_context(
        self, 
        document_content: List[Dict[str, Any]], 
        document_analysis: Optional[Dict[str, Any]] = None,
        max_chunks: int = 5
    ) -> Dict[str, Any]:
        """Prepare context for prompt enhancement"""
        context = {}
        
        # Add document analysis if available
        if document_analysis:
            context["document_analysis"] = document_analysis
        
        # Add sample content from the document (limit to max_chunks)
        if document_content:
            sample_chunks = document_content[:max_chunks]
            context["document_samples"] = sample_chunks
            
            # Add document stats
            context["document_stats"] = {
                "total_chunks": len(document_content),
                "sample_count": len(sample_chunks),
                "page_count": max([chunk.get("page_number", 0) for chunk in document_content], default=0)
            }
        
        return context

def example_pdf_optimization():
    """Example of optimizing a prompt for a PDF document"""
    # You'll need to replace this with an actual PDF file path
    pdf_file = "path/to/your/document.pdf"
    
    # Only run this example if the file exists
    if not os.path.exists(pdf_file):
        print(f"PDF file not found: {pdf_file}")
        print("Skipping PDF optimization example.")
        return
    
    print("\n=== Example: PDF Document Prompt Optimization ===")
    
    # Base prompt for document analysis
    base_prompt = """
    Analyze the document and extract the key information.
    """
    
    # New guidelines from user
    new_guidelines = """
    I need to focus on the financial metrics, risk assessments, and market predictions.
    Please extract specific numeric data and organize it by business segments.
    Also identify any warnings or cautionary statements in the document.
    """
    
    # Create optimizer
    optimizer = DocumentPromptOptimizer(client_type="azure")
    
    # Optimize prompt for the document
    result = optimizer.optimize_for_document(
        file_path=pdf_file,
        base_prompt=base_prompt,
        new_guidelines=new_guidelines,
        enhancement_type="document"
    )
    
    # Display results
    if result.get("success", False):
        print("\nDocument Analysis Summary:")
        analysis = result.get("document_analysis", {})
        print(f"Document Type: {analysis.get('document_type', 'Unknown')}")
        print(f"Complexity: {analysis.get('complexity_level', 'Unknown')}")
        print(f"Target Audience: {analysis.get('target_audience', 'Unknown')}")
        
        print("\nOptimized Prompt:")
        print(result.get("enhanced_prompt"))
        
        # Save result to file
        with open("pdf_optimized_prompt.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\nSaved full results to: pdf_optimized_prompt.json")
    else:
        print(f"\nOptimization failed: {result.get('error', 'Unknown error')}")

def example_custom_optimization():
    """Example with manually provided document content (when no file available)"""
    print("\n=== Example: Custom Document Prompt Optimization ===")
    
    # Base prompt
    base_prompt = """
    Summarize the key points from the quarterly report.
    """
    
    # New guidelines
    new_guidelines = """
    I'm interested in competitor analysis and market positioning.
    Focus on how the company compares to others in the industry and what
    strategic actions they're taking to improve market share.
    """
    
    # Manually create sample content
    sample_content = [
        {
            "text": """
            Q2 QUARTERLY REPORT - TECHCORP INC.
            
            EXECUTIVE SUMMARY
            TechCorp achieved record revenue of $1.8B in Q2 2024, representing 15% YoY growth.
            Operating margin expanded to 28%, exceeding analyst expectations of 26%.
            Cloud services division continues to be our fastest-growing segment at 32% YoY.
            """,
            "page_number": 1
        },
        {
            "text": """
            MARKET POSITION
            TechCorp maintained its #2 position in enterprise cloud solutions with 23% market share.
            Main competitor CloudLeader holds 27% share but growth has slowed to 14% (vs our 32%).
            New entrant RapidTech captured 5% share, focusing primarily on SMB customers.
            """,
            "page_number": 2
        },
        {
            "text": """
            COMPETITIVE STRATEGY
            1. Enterprise Focus: Increasing investment in enterprise sales team by 40%
            2. Product Differentiation: Launching AI-powered analytics platform in Q3
            3. Strategic Partnerships: New alliance with NetSystems to expand distribution
            4. Talent Acquisition: Hired 28 engineers from competitors in key AI/ML roles
            """,
            "page_number": 3
        }
    ]
    
    # Create enhancer directly (simplified approach)
    enhancer = PromptEnhancer(client_type="azure")
    
    # Enhance the prompt with provided content
    result = enhancer.enhance_prompt(
        base_prompt=base_prompt,
        new_prompt=new_guidelines,
        context_data=sample_content,
        enhancement_type="document",
        temperature=0.3
    )
    
    # Display results
    print("\nEnhanced Prompt:")
    print(result.get("enhanced_prompt"))
    
    print("\nExplanation:")
    print(result.get("explanation"))

if __name__ == "__main__":
    # Run the examples
    try:
        # Try the PDF example if file exists
        example_pdf_optimization()
        
        # Custom example that doesn't require file
        example_custom_optimization()
    except Exception as e:
        print(f"Error running examples: {str(e)}")