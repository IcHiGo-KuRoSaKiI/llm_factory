import os
from typing import Dict, Type

from .base_parser import BaseParser
from .pdf.parser import PDFParser
from .docx.parser import DocxParser
from .pptx.parser import PPTProcessor


class ParserFactory:
    """
    Factory class for creating different types of document parsers based on file extension.
    """
    
    # Registry mapping file extensions to parser classes
    _parsers: Dict[str, Type[BaseParser]] = {
        '.pdf': PDFParser,
        '.docx': DocxParser,
        '.doc': DocxParser,  # Same parser for .doc files
        '.pptx': PPTProcessor,
        '.ppt': PPTProcessor,  # Same processor for .ppt files
    }
    
    @classmethod
    def create_parser(cls, file_path: str, openai_helper, ingester_logger=None) -> BaseParser:
        """
        Create and return an appropriate parser for the given file.
        
        Args:
            file_path: Path to the document file
            openai_helper: Helper for OpenAI API operations
            ingester_logger: Optional logger for ingestion operations
            
        Returns:
            An instance of the appropriate parser for the file type
            
        Raises:
            ValueError: If the file extension is not supported
        """
        # Get the file extension (lowercase)
        _, ext = os.path.splitext(file_path.lower())
        
        # Check if the extension is supported
        if ext not in cls._parsers:
            supported_extensions = ', '.join(cls._parsers.keys())
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported extensions are: {supported_extensions}"
            )
        
        # Create and return the appropriate parser
        parser_class = cls._parsers[ext]
        return parser_class(openai_helper, ingester_logger)
    
    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[BaseParser]) -> None:
        """
        Register a new parser for a specific file extension.
        
        Args:
            extension: File extension to register (including the dot, e.g., '.xlsx')
            parser_class: Parser class to use for this extension
        """
        cls._parsers[extension.lower()] = parser_class
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        Get a list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(cls._parsers.keys())