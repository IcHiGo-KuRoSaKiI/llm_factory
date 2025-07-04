# parsers/factory.py (modified)
import os
import logging
from typing import Dict, Type, List

from .base_parser import BaseParser
from .pdf.parser import PDFParser
from .docx.parser import DocxParser
from .pptx.parser import PPTProcessor
from .pptx.lightweight_parser import LightweightPPTParser
from .pptx.pdf_bridge_parser import PDFBridgePPTParser

logger = logging.getLogger(__name__)


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

    # Fallback parsers for when the primary parser fails (in order of preference)
    _fallback_parsers: Dict[str, List[Type[BaseParser]]] = {
        '.pptx': [LightweightPPTParser, PDFBridgePPTParser],
        '.ppt': [LightweightPPTParser, PDFBridgePPTParser],
    }

    @classmethod
    def create_parser(cls, file_path: str, openai_helper, ingester_logger=None) -> BaseParser:
        """
        Create and return an appropriate parser for the given file.
        Will attempt to use fallback parsers if the primary parser fails.

        Args:
            file_path: Path to the document file
            openai_helper: Helper for OpenAI API operations
            ingester_logger: Optional logger for ingestion operations

        Returns:
            An instance of the appropriate parser for the file type

        Raises:
            ValueError: If the file extension is not supported or all parsers fail
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

        # First try the primary parser
        parser_class = cls._parsers[ext]
        try:
            logger.info(
                f"Creating primary parser {parser_class.__name__} for {ext} file")
            return parser_class(openai_helper, ingester_logger)
        except Exception as primary_error:
            # If there are fallback parsers for this extension, try them in order
            if ext in cls._fallback_parsers and cls._fallback_parsers[ext]:
                logger.warning(
                    f"Primary parser failed for {ext}: {str(primary_error)}. Trying fallback parsers.")

                errors = [
                    f"Primary parser ({parser_class.__name__}) error: {str(primary_error)}"]

                # Try each fallback parser in order
                for i, fallback_class in enumerate(cls._fallback_parsers[ext], 1):
                    try:
                        logger.info(
                            f"Creating fallback parser #{i} ({fallback_class.__name__}) for {ext} file")
                        return fallback_class(openai_helper, ingester_logger)
                    except Exception as fallback_error:
                        error_msg = f"Fallback #{i} ({fallback_class.__name__}) error: {str(fallback_error)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                # If we get here, all parsers failed
                raise ValueError(
                    f"All parsers failed for {ext} file:\n" + "\n".join(errors))
            else:
                # Re-raise the exception if no fallback is available
                raise

    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[BaseParser], is_fallback: bool = False) -> None:
        """
        Register a new parser for a specific file extension.

        Args:
            extension: File extension to register (including the dot, e.g., '.xlsx')
            parser_class: Parser class to use for this extension
            is_fallback: Whether this is a fallback parser (default: False)
        """
        if is_fallback:
            if extension.lower() not in cls._fallback_parsers:
                cls._fallback_parsers[extension.lower()] = []
            cls._fallback_parsers[extension.lower()].append(parser_class)
        else:
            cls._parsers[extension.lower()] = parser_class

    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        Get a list of all supported file extensions.

        Returns:
            List of supported file extensions
        """
        return list(cls._parsers.keys())
