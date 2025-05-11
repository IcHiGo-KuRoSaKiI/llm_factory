import base64
import re
from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.text.paragraph import Paragraph
import io
from PIL import Image
from typing import List, Dict, Any, Optional
import os

from ..base_parser import BaseParser

class DocxParser(BaseParser):
    """
    Parser for DOCX documents that extracts text, images, and tables.
    """
    
    def __init__(self, openai_helper, ingester_logger=None):
        """
        Initialize the DOCX parser with dependencies.
        
        Args:
            openai_helper: Helper for OpenAI API calls
            ingester_logger: Optional logger for ingestion operations
        """
        super().__init__(openai_helper, ingester_logger)
        self.current_page = 1
        self.words_per_page = 500  # Approximate words per page
        self.current_word_count = 0
        self.current_chunk_content = []
        self.chunks = []
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the DOCX document at the specified file path.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of dictionaries containing extracted content with page numbers
        """
        return self.parse_docx(file_path)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unnecessary formatting and standardizing punctuation."""
        # Replace multiple dots with single ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Clean unnecessary line breaks while preserving paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'\n', ' ', text)  # Replace single line breaks with space
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Normalize spaces after punctuation
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # Fix spacing around parentheses and brackets
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        text = re.sub(r'\s*\[\s*', ' [', text)
        text = re.sub(r'\s*\]\s*', '] ', text)
        
        # Remove tabs
        text = text.replace('\t', ' ')
        
        # Remove index markers (e.g., "....1", "...2")
        text = re.sub(r'\.{2,}\s*\d+', '', text)
        
        # Fix multiple consecutive spaces again (in case previous operations created any)
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text

    def encode_image_from_blob(self, image_blob: bytes) -> str:
        """Encode image blob directly to base64 without saving to disk."""
        return base64.b64encode(image_blob).decode("utf-8")

    def process_image(self, image_blob: bytes) -> str:
        """Process image through OpenAI Vision API."""
        try:
            base64_encoded = self.encode_image_from_blob(image_blob)
            data_uri_png = f"data:image/png;base64,{base64_encoded}"
            return self.openai_helper.get_openai_response_image(data_uri_png)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "[Image processing failed]"

    def get_heading_level(self, paragraph: Paragraph) -> Optional[int]:
        """Get the heading level of a paragraph (None if not a heading)."""
        if paragraph.style.name.startswith('Heading'):
            try:
                return int(paragraph.style.name[-1])
            except ValueError:
                return None
        return None

    def create_new_chunk(self):
        """Create a new chunk and reset the word counter."""
        if self.current_chunk_content:
            joined_text = "\n".join(self.current_chunk_content)
            cleaned_text = self.clean_text(joined_text)
            self.chunks.append({
                "text": cleaned_text,
                "page_number": self.current_page
            })
            self.current_chunk_content = []
            self.current_word_count = 0
            self.current_page += 1

    def process_table(self, table: Table) -> str:
        """Convert a table to text representation."""
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                # Process the text content of the cell
                cell_text = " ".join(paragraph.text for paragraph in cell.paragraphs)
                row_data.append(cell_text)
            table_data.append(" | ".join(row_data))
        return "\n".join(table_data)

    def iter_block_items(self, parent):
        """Yield each paragraph and table child within the document, in document order."""
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("Something's not right")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse DOCX file and return structured content with images and headings.
        """
        try:
            print(f"\n📑 Processing DOCX file: {os.path.basename(file_path)}")
            doc = Document(file_path)
            current_heading = None
            current_subheading = None

            for block in self.iter_block_items(doc):
                if isinstance(block, Paragraph):
                    heading_level = self.get_heading_level(block)
                    
                    # Process headings
                    if heading_level is not None:
                        if heading_level == 1:
                            if self.current_word_count > 0:
                                self.create_new_chunk()
                            current_heading = block.text
                            self.current_chunk_content.append(f"\n# {block.text}\n")
                        elif heading_level == 2:
                            current_subheading = block.text
                            self.current_chunk_content.append(f"\n## {block.text}\n")
                        else:
                            self.current_chunk_content.append(f"\n{'#' * heading_level} {block.text}\n")
                    
                    # Process regular paragraphs
                    else:
                        # Add paragraph text
                        if block.text.strip():
                            self.current_chunk_content.append(block.text)
                            self.current_word_count += len(block.text.split())

                        # Process images in the paragraph
                        for run in block.runs:
                            image_found = False
                            image_blob = None
                            
                            # Handle images in drawings/shapes
                            for drawing in run._element.findall(".//w:drawing", {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                                try:
                                    print("Found drawing element")
                                    inline_or_anchor = drawing.find(".//wp:inline", {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'})
                                    if inline_or_anchor is None:
                                        inline_or_anchor = drawing.find(".//wp:anchor", {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'})
                                    
                                    if inline_or_anchor is not None:
                                        print("Found inline/anchor element")
                                        graphic = inline_or_anchor.find(".//a:graphic", {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                                        if graphic is not None:
                                            graphicData = graphic.find(".//a:graphicData", {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                                            if graphicData is not None:
                                                pic = graphicData.find(".//pic:pic", {'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'})
                                                if pic is not None:
                                                    blip = pic.find(".//a:blip", {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                                                    if blip is not None:
                                                        image_rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                                        if image_rid:
                                                            print(f"Found image with relationship ID: {image_rid}")
                                                            image_part = doc.part.related_parts[image_rid]
                                                            image_blob = image_part.blob
                                                            image_found = True
                                except Exception as e:
                                    error_msg = f"Error processing an image: {str(e)}"
                                    print(f"⚠️ {error_msg}")
                                    
                                    if self.ingester_logger:
                                        # Log the image processing failure
                                        self.ingester_logger.log_failure(
                                            file_path=file_path,
                                            error=error_msg,
                                            error_type="IMAGE_PROCESSING_ERROR"
                                        )
                                    continue

                            # Handle direct image elements
                            if not image_found:
                                for picture in run._element.findall(".//w:pict", {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                                    try:
                                        print("Found picture element")
                                        shape = picture.find(".//v:shape", {'v': 'urn:schemas-microsoft-com:vml'})
                                        if shape is not None:
                                            imagedata = shape.find(".//v:imagedata", {'v': 'urn:schemas-microsoft-com:vml'})
                                            if imagedata is not None:
                                                image_rid = imagedata.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}href')
                                                if image_rid:
                                                    print(f"Found image with relationship ID: {image_rid}")
                                                    image_part = doc.part.related_parts[image_rid]
                                                    image_blob = image_part.blob
                                                    image_found = True
                                    except Exception as e:
                                        print(f"Warning: Error processing a picture: {str(e)}")
                                        continue

                            # Process the image if found
                            if image_found and image_blob:
                                try:
                                    print("Processing found image...")
                                    # Process image and add its text description
                                    image_text = self.process_image(image_blob)
                                    if image_text:
                                        print(f"Image text extracted: {image_text[:100]}...")  # Print first 100 chars
                                        self.current_chunk_content.append(f"\n[Image Description: {image_text}]\n")
                                except Exception as e:
                                    print(f"Error processing image content: {str(e)}")

                elif isinstance(block, Table):
                    # Process tables
                    table_text = self.process_table(block)
                    self.current_chunk_content.append(f"\n{table_text}\n")
                    self.current_word_count += len(table_text.split())

                # Check if we need to create a new chunk based on word count
                if self.current_word_count >= self.words_per_page:
                    self.create_new_chunk()

            # Add any remaining content as the last chunk
            if self.current_chunk_content:
                self.create_new_chunk()

            # Log successful processing
            if self.chunks:
                print(f"\n✅ Successfully processed {os.path.basename(file_path)}")
                print(f"   Generated {len(self.chunks)} chunks")
                self.log_success(file_path)

            return self.chunks

        except Exception as e:
            error_msg = f"Error parsing DOCX file: {str(e)}"
            print(f"❌ {error_msg}")
            self.log_failure(file_path, str(e), "DOCX_PARSING_ERROR")
            raise Exception(error_msg)