# utils/image_utils.py
import os
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 format
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string or None if encoding fails
    """
    try:
        if not os.path.isfile(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

def convert_to_data_uri(base64_string: str, mime_type: str = "image/jpeg") -> str:
    """
    Convert a base64 string to a data URI
    
    Args:
        base64_string: Base64-encoded string
        mime_type: MIME type of the image (default: image/jpeg)
        
    Returns:
        Data URI string
    """
    if base64_string.startswith("data:"):
        return base64_string
    return f"data:{mime_type};base64,{base64_string}"

def image_to_data_uri(image_path: str, mime_type: Optional[str] = None) -> Optional[str]:
    """
    Convert an image file to a data URI
    
    Args:
        image_path: Path to the image file
        mime_type: Optional MIME type of the image (auto-detected if not provided)
        
    Returns:
        Data URI string or None if conversion fails
    """
    try:
        # Detect MIME type from file extension if not provided
        if not mime_type:
            extension = os.path.splitext(image_path)[1].lower()
            if extension in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif extension == '.png':
                mime_type = "image/png"
            elif extension == '.gif':
                mime_type = "image/gif"
            elif extension == '.webp':
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default to JPEG
        
        # Encode image to base64 and convert to data URI
        base64_encoded = encode_image_to_base64(image_path)
        if base64_encoded:
            return convert_to_data_uri(base64_encoded, mime_type)
        return None
        
    except Exception as e:
        logger.error(f"Error converting image to data URI: {str(e)}")
        return None