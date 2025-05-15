# parsers/pptx/lightweight_parser.py
import os
import tempfile
import shutil
from io import BytesIO
import base64
from typing import List, Dict, Any

from ..base_parser import BaseParser


class LightweightPPTParser(BaseParser):
    """
    Lightweight parser for PowerPoint (PPT/PPTX) files that converts slides to images
    without using Aspose.slides. Uses python-pptx, pptx2pdf and pdf2image for cross-platform
    compatibility.
    """

    def __init__(self, openai_helper, ingester_logger=None):
        """
        Initialize the lightweight PowerPoint parser with dependencies.

        Args:
            openai_helper: Helper for OpenAI API calls for image processing
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
        temp_dir = None
        try:
            # Import dependencies here to avoid forcing installation if not used
            try:
                from pptx import Presentation
                from pdf2image import convert_from_path
                import matplotlib.pyplot as plt
                from PIL import Image
            except ImportError as e:
                missing_lib = str(e).split(
                    "'")[-2] if "'" in str(e) else str(e)
                raise ImportError(f"Required package not installed: {missing_lib}. "
                                  f"Install with 'pip install python-pptx pdf2image pillow matplotlib'")

            print(
                f"\n📊 Processing PowerPoint file using lightweight image-based parser: {os.path.basename(file_path)}")

            # Create a temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            print(f"   ✓ Created temporary directory: {temp_dir}")

            # Step 1: Convert slides to images using a two-step process
            # First render slides using python-pptx + matplotlib, then process images with OpenAI

            # Load the presentation
            prs = Presentation(file_path)
            print(
                f"   ✓ Successfully loaded presentation with {len(prs.slides)} slides")

            all_text = []

            # Process slides
            for slide_index, slide in enumerate(prs.slides, 1):
                print(f"   • Processing slide {slide_index}/{len(prs.slides)}")
                try:
                    # Save slide as image
                    img_path = os.path.join(
                        temp_dir, f"slide_{slide_index}.png")

                    # Approach 1: Render slide using matplotlib (a basic approach)
                    # This won't be perfect but should work across platforms
                    width_inches = 10  # 10 inches wide
                    height_inches = 7.5  # 7.5 inches high (4:3 aspect ratio)
                    dpi = 150  # Higher DPI for better resolution

                    # Create a figure with the right size
                    fig, ax = plt.subplots(
                        figsize=(width_inches, height_inches), dpi=dpi)
                    ax.set_position([0, 0, 1, 1])  # Fill the entire figure

                    # Extract individual elements for better rendering
                    slide_elements = []

                    # Add title
                    if slide.shapes.title and slide.shapes.title.text:
                        title_text = slide.shapes.title.text
                        slide_elements.append({
                            'type': 'title',
                            'text': title_text,
                            # Centered horizontally, near top
                            'position': (0.5, 0.1)
                        })

                    # Add other text elements
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text and shape != slide.shapes.title:
                            # Skip the title which we already processed
                            slide_elements.append({
                                'type': 'text',
                                'text': shape.text,
                                'position': (0.5, 0.5)  # Centered
                            })

                    # Render elements
                    ax.axis('off')  # Turn off axis

                    # Add a basic white background
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                               facecolor='white', edgecolor='none', zorder=-1))

                    # Render title
                    title_elements = [
                        el for el in slide_elements if el['type'] == 'title']
                    for el in title_elements:
                        ax.text(el['position'][0], el['position'][1], el['text'],
                                fontsize=18, fontweight='bold', ha='center', va='center')

                    # Render other text
                    text_elements = [
                        el for el in slide_elements if el['type'] == 'text']

                    # Distribute text vertically
                    text_y_start = 0.25  # Start below title
                    # Distribute with max step size
                    text_y_step = min(0.6 / max(len(text_elements), 1), 0.15)

                    for i, el in enumerate(text_elements):
                        ax.text(el['position'][0], text_y_start + i * text_y_step, el['text'],
                                fontsize=12, ha='center', va='center', wrap=True)

                    # Add slide number
                    ax.text(
                        0.95, 0.95, f"Slide {slide_index}", fontsize=8, ha='right', va='top')

                    # Save figure to file
                    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    # Now process the rendered image with OpenAI Vision
                    with open(img_path, "rb") as img_file:
                        base64_encoded = base64.b64encode(
                            img_file.read()).decode('utf-8')
                        data_uri = f"data:image/png;base64,{base64_encoded}"

                    # Extract text using the OpenAI vision API
                    prompt = """
                    Extract all text content from this PowerPoint slide. 
                    Describe any diagrams, charts, or tables you see.
                    Preserve the structure of the content as much as possible.
                    """

                    extracted_text = self.openai_helper.get_openai_response_image(
                        data_uri, prompt=prompt
                    )

                    all_text.append({
                        "text": extracted_text,
                        "page_number": slide_index
                    })

                    print(f"   ✓ Processed slide {slide_index}")

                except Exception as slide_error:
                    print(
                        f"   ⚠️ Error processing slide {slide_index}: {str(slide_error)}")
                    all_text.append({
                        "text": f"[Error processing slide {slide_index}: {str(slide_error)}]",
                        "page_number": slide_index
                    })

            # Log success
            print(
                f"   ✅ Successfully processed {len(prs.slides)} slides using lightweight parser")
            self.log_success(file_path)

            return all_text

        except Exception as e:
            error_msg = f"Error parsing PPT file with lightweight parser: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.log_failure(file_path, str(
                e), "LIGHTWEIGHT_PPT_PARSING_ERROR")
            raise

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("   ✓ Cleaned up temporary files")
