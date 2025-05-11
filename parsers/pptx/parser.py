from PIL import Image
import tkinter as tk
from tkinter import ttk
import os
import tempfile
import shutil
import traceback
import aspose.slides as slides
from pdf2image import convert_from_path
from PIL import Image, ImageTk
import base64
from typing import List, Dict, Any

from ..base_parser import BaseParser

class PPTProcessor(BaseParser):
    """
    Processor for PowerPoint (PPT/PPTX) files that converts slides to images
    and extracts text content using OCR via OpenAI's Vision API.
    """
    
    def __init__(self, openai_helper, ingester_logger=None):
        """
        Initialize the PowerPoint processor with dependencies.
        
        Args:
            openai_helper: Helper for OpenAI API calls
            ingester_logger: Optional logger for ingestion operations
        """
        super().__init__(openai_helper, ingester_logger)
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the PowerPoint document at the specified file path.
        
        Args:
            file_path: Path to the PPT/PPTX file
            
        Returns:
            List of dictionaries containing extracted content with page numbers
        """
        return self.process_ppt(file_path)
    
    def encode_image(self, image_path):
        """Encode an image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def process_ppt(self, file_path, debug_view=False):
        """Convert PPT/PPTX to images with enhanced error handling using Aspose.Slides"""
        print(f"\n📊 Processing PowerPoint file: {os.path.basename(file_path)}")
        temp_dir = None
        original_file_name = os.path.basename(file_path)

        def show_image(image_path, slide_num, total_slides):
            """Helper function to display image in a window"""
            # Create and configure root window
            root = tk.Tk()
            root.title(f"Slide {slide_num}/{total_slides}")
            
            # Add a label with instructions
            label = ttk.Label(root, text="Press 'q' to close and continue processing")
            label.pack(pady=5)
            
            # Load and display image
            img = Image.open(image_path)
            # Resize if too large while maintaining aspect ratio
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Create canvas and display image
            canvas = tk.Canvas(root, width=img.size[0], height=img.size[1])
            canvas.pack()
            canvas.create_image(0, 0, anchor="nw", image=photo)
            
            def close_window(event=None):
                if event and event.char != 'q':
                    return
                root.destroy()
                
            # Bind 'q' key to close window
            root.bind('<Key>', close_window)
            
            # Center window on screen
            root.update()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x = (screen_width - root.winfo_width()) // 2
            y = (screen_height - root.winfo_height()) // 2
            root.geometry(f"+{x}+{y}")
            
            root.mainloop()

        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            print(f"   ✓ Created temporary directory: {temp_dir}")

            # Create a temporary copy of the input file
            temp_input_file = os.path.join(temp_dir, original_file_name)
            shutil.copy2(file_path, temp_input_file)
            print(f"   ✓ Created temporary input file: {temp_input_file}")

            # Convert PowerPoint to PDF using Aspose.Slides
            print(f"   • Converting PowerPoint to PDF using Aspose.Slides...")
            try:
                with slides.Presentation(temp_input_file) as presentation:
                    pdf_path = os.path.join(temp_dir, "output.pdf")
                    presentation.save(pdf_path, slides.export.SaveFormat.PDF)
                print(f"   ✓ PDF conversion complete: {pdf_path}")
            except Exception as e:
                raise RuntimeError(f"Error during PowerPoint to PDF conversion: {str(e)}")

            # Verify the PDF file was created
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                raise ValueError("Generated PDF is empty or not created.")

            # Convert PDF to images
            print("   • Converting PDF to images...")
            try:
                # Note: You may need to adjust the poppler_path based on your environment
                images = convert_from_path(
                    pdf_path,
                    output_folder=temp_dir,
                    fmt="png",
                    poppler_path=r"C:\Users\VEDANSH.KUMAR\Downloads\poppler-24.08.0\Library\bin"
                )
                print(f"   ✓ Converted {len(images)} pages to images")

                # Process each image
                all_text = []
                for i, image in enumerate(images, 1):
                    image_path = os.path.join(temp_dir, f"slide_{i}.png")
                    image.save(image_path, 'PNG')

                    # Show image if debug_view is True
                    if debug_view:
                        show_image(image_path, i, len(images))

                    base64_encoded_image = self.encode_image(image_path)
                    data_uri_png = f"data:image/png;base64,{base64_encoded_image}"

                    try:
                        content = self.openai_helper.get_openai_response_image(data_uri_png)
                        all_text.append({
                            "text": content,
                            "page_number": i
                        })
                        print(f"   ✓ Processed slide {i}/{len(images)}")
                    except Exception as e:
                        print(f"   ⚠️ Error processing slide {i}: {str(e)}")
                        continue

                    # Clean up image
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        print(f"   ⚠️ Warning: Could not remove temp image: {str(e)}")

                # Log success
                self.log_success(file_path)
                
                return all_text
            except Exception as e:
                print(f"   ❌ PDF to image conversion failed: {str(e)}")
                raise

        except Exception as e:
            print(f"\n❌ PowerPoint processing failed: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")

            self.log_failure(
                file_path=file_path,
                error=str(e),
                error_type="PPT_PROCESSING_FAILURE"
            )
                
            raise

        finally:
            # Clean up the temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print("   ✓ Cleaned up temporary files")
                except Exception as e:
                    print(f"   ⚠️ Cleanup warning: {str(e)}")