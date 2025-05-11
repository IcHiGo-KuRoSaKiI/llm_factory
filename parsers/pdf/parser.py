import os
import fitz
import base64
import json
from datetime import datetime
from PIL import Image
import io
from typing import List, Dict, Any

from ..base_parser import BaseParser
from .chunking import process_document_chunks

class PDFParser(BaseParser):
    """
    Parser for PDF documents that extracts text, images, and tables.
    """
    
    def __init__(self, openai_helper, ingester_logger=None):
        """
        Initialize the PDF parser with dependencies.
        
        Args:
            openai_helper: Helper for OpenAI API calls
            ingester_logger: Optional logger for ingestion operations
        """
        super().__init__(openai_helper, ingester_logger)
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the PDF document at the specified file path.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing extracted content with page numbers
        """
        return self.extract_text_and_images_from_pdf(file_path)
    
    def get_sort_key(self, item):
        """
        Custom sorting function that handles different coordinate systems:
        - For images: uses x0, y0
        - For text: uses x1, y1 
        - For comparison between types: compares appropriate coordinates
        """
        pos = item["position"]
        content_type = item["type"]
        
        if content_type == "image":
            x = pos["x0"]
            y = pos["y0"]
        else:  # text
            x = pos["x1"]
            y = pos["y1"]
            
        # Return tuple for sorting - first by y (vertical), then by x (horizontal)
        return (y, x)

    def sort_page_content(self, content):
        """Sort page content by position"""
        import functools
        
        def compare_positions(item1, item2):
            pos1 = item1["position"]
            pos2 = item2["position"]
            type1 = item1["type"]
            type2 = item2["type"]
            # Get y coordinates
            y1 = pos1["y0"] if type1 == "image" else pos1["y1"]
            y2 = pos2["y0"] if type2 == "image" else pos2["y1"]
            
            # Get x coordinates
            x1 = pos1["x0"] if type1 == "image" else pos1["x1"]
            x2 = pos2["x0"] if type2 == "image" else pos2["x1"]
            
            # First compare y coordinates (top to bottom)
            y_diff = y1 - y2
            if abs(y_diff) > 5:  # Using small threshold to group items on similar vertical positions
                return -1 if y_diff < 0 else 1
                
            # If y coordinates are similar, compare x coordinates (left to right)
            return -1 if x1 < x2 else 1 if x1 > x2 else 0
        
        # Use Python's sorted with custom comparison function
        return sorted(content, key=functools.cmp_to_key(compare_positions))

    def extract_text_from_chunks(self, processed_document):
        """
        Extract text and page numbers from processed document's chunks.
        
        Args:
            processed_document: Dictionary containing the processed document with chunks
            
        Returns:
            List of dictionaries with text and page number
        """
        all_text = []
        
        # Get chunks from the processed document
        chunks = processed_document.get('chunks', [])
        
        # Extract text and page number from each chunk
        for chunk in chunks:
            all_text.append({
                "text": chunk["text"],
                "page_number": chunk["page_number"]
            })
        
        return all_text

    def extract_text_and_images_from_pdf(self, pdf_path):
        """Extract text and images from PDF, process content, and chunk"""
        try:
            # Create TableProcessor for table extraction (simplified for this example)
            class TableProcessor:
                def __init__(self, openai_helper):
                    self.openai_helper = openai_helper
                
                def smart_table_process(self, pdf_path, keep_text=True):
                    # Placeholder for table extraction - in a real scenario, this would use the actual implementation
                    return []
            
            table_processor = TableProcessor(self.openai_helper)
            
            # First, extract content
            content_data = []
            pdf_document = fitz.open(pdf_path)
            
            print(f"Processing PDF with {pdf_document.page_count} pages...")
            
            try:
                for page_num in range(len(pdf_document)):
                    try:
                        print(f"\nProcessing page {page_num + 1}...")
                        page = pdf_document[page_num]
                        page_content = {
                            "page_number": page_num + 1,
                            "content": []
                        }

                        # Extract text blocks
                        try:
                            text_blocks = page.get_text("blocks")
                            for block in text_blocks:
                                text = block[4]  # Extract text from block
                                if text.strip():  # Only include non-empty text
                                    page_content["content"].append({
                                        "type": "text",
                                        "content": text.strip(),
                                        "position": {
                                            "x1": block[0],
                                            "y1": block[1],
                                            "x2": block[2],
                                            "y2": block[3]
                                        }
                                    })
                        except Exception as e:
                            error_msg = f"Error extracting text blocks on page {page_num + 1}: {str(e)}"
                            print(error_msg)
                            self.log_failure(pdf_path, error_msg, "TEXT_BLOCK_EXTRACTION_FAILURE")

                        # Extract images
                        try:
                            image_list = page.get_images(full=True)
                            for img_index, img in enumerate(image_list):
                                try:
                                    xref = img[0]
                                    
                                    # Get image bounding boxes
                                    masks = page.get_image_rects(xref)
                                    print(f"Image {img_index} masks on page {page_num + 1}:", masks)
                                    
                                    if not masks:
                                        print(f"No bounding boxes found for image {img_index} on page {page_num + 1}")
                                        continue
                                    
                                    # Extract image data and directly encode it
                                    base_image = pdf_document.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    
                                    # Directly encode image bytes to base64
                                    base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                                    data_uri_png = f"data:image/png;base64,{base64_encoded_image}"
                                    img_text = self.openai_helper.get_openai_response_image(data_uri_png)

                                    for bbox in masks:
                                        page_content["content"].append({
                                            "type": "image",
                                            "extracted_text": img_text,
                                            "position": {
                                                "x0": bbox.x0,
                                                "y0": bbox.y0,
                                                "x1": bbox.x1,
                                                "y1": bbox.y1,
                                                "width": bbox.width,
                                                "height": bbox.height
                                            }
                                        })

                                except Exception as e:
                                    error_msg = f"Failed to process image {img_index} on page {page_num + 1}: {str(e)}"
                                    print(error_msg)
                                    self.log_failure(pdf_path, error_msg, "IMAGE_PROCESSING_FAILURE")
                                    continue
                        except Exception as e:
                            error_msg = f"Error processing images on page {page_num + 1}: {str(e)}"
                            print(error_msg)
                            self.log_failure(pdf_path, error_msg, "IMAGE_EXTRACTION_FAILURE")

                        # Sort content by vertical position
                        page_content["content"] = self.sort_page_content(page_content["content"])
                        content_data.append(page_content)

                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        print(error_msg)
                        self.log_failure(pdf_path, error_msg, "PAGE_PROCESSING_FAILURE")
            finally:
                pdf_document.close()

            # Try to extract tables
            try:
                table_content = table_processor.smart_table_process(pdf_path, keep_text=False)
                
                if table_content:
                    combined_content = []
                    
                    # Convert existing content
                    for page in content_data:
                        for item in page['content']:
                            item_with_page = item.copy()
                            item_with_page['page'] = str(page['page_number'])
                            combined_content.append(item_with_page)
                    
                    combined_content.extend(table_content)
                    combined_content = sorted(combined_content, key=lambda k: (k['page'], k.get('y1', 0), k.get('x1', 0)))
                    
                    # Reconstruct content_data
                    content_data = []
                    current_page = None
                    current_page_content = None
                    
                    for item in combined_content:
                        page_num = item['page']
                        if page_num != current_page:
                            if current_page_content:
                                content_data.append(current_page_content)
                            current_page = page_num
                            current_page_content = {
                                "page_number": int(page_num),
                                "content": []
                            }
                        
                        item_to_add = item.copy()
                        item_to_add.pop('page', None)
                        current_page_content['content'].append(item_to_add)
                    
                    if current_page_content:
                        content_data.append(current_page_content)

            except Exception as e:
                error_msg = f"Error processing tables: {str(e)}"
                print(error_msg)
                self.log_failure(pdf_path, error_msg, "TABLE_PROCESSING_FAILURE")

            # Prepare final JSON structure
            output_data = {
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "processed_date": datetime.now().isoformat(),
                    "total_pages": len(content_data)
                },
                "pages": content_data
            }
            
            # Process the document
            processed_document = process_document_chunks(output_data)
            
            # Extract text from chunks
            extracted_text = self.extract_text_from_chunks(processed_document)
            
            # Log success
            self.log_success(pdf_path)
            
            return extracted_text

        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            print(error_msg)
            self.log_failure(pdf_path, error_msg, "PDF_PROCESSING_FAILURE")
            return []