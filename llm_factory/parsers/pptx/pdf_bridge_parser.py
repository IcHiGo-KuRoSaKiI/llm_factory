# parsers/pptx/pdf_bridge_parser.py
import os
import tempfile
import shutil
import base64
from typing import List, Dict, Any

from ..base_parser import BaseParser


class PDFBridgePPTParser(BaseParser):
    """
    A PowerPoint parser that first converts to PDF, then converts PDF pages to images.
    This provides a second fallback option if other methods fail.
    """

    def __init__(self, openai_helper, ingester_logger=None):
        """
        Initialize the PDF-bridge PowerPoint parser with dependencies.

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
            print(
                f"\n📊 Processing PowerPoint file using PDF bridge parser: {os.path.basename(file_path)}")

            # Create a temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            print(f"   ✓ Created temporary directory: {temp_dir}")

            # Convert PPTX to PDF using python-pptx-export
            pdf_path = os.path.join(temp_dir, "presentation.pdf")

            try:
                # Try the first method - using comtypes on Windows
                self._convert_with_comtypes(file_path, pdf_path)
            except Exception as e1:
                print(f"   ⚠️ comtypes conversion failed: {str(e1)}")

                try:
                    # Try the second method - using LibreOffice (if available)
                    self._convert_with_libreoffice(file_path, pdf_path)
                except Exception as e2:
                    print(f"   ⚠️ LibreOffice conversion failed: {str(e2)}")

                    try:
                        # Try the third method - using unoconv (if available)
                        self._convert_with_unoconv(file_path, pdf_path)
                    except Exception as e3:
                        print(f"   ⚠️ unoconv conversion failed: {str(e3)}")
                        raise ValueError(
                            "Failed to convert PPTX to PDF using any available method")

            # Check if PDF was created successfully
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                raise ValueError(
                    "PDF conversion failed - output file is empty or not created")

            print(f"   ✓ Successfully converted to PDF: {pdf_path}")

            # Convert PDF to images using pdf2image
            from pdf2image import convert_from_path

            print("   • Converting PDF to images...")
            images = convert_from_path(
                pdf_path,
                output_folder=temp_dir,
                fmt="png",
                dpi=200
            )
            print(f"   ✓ Converted {len(images)} pages to images")

            # Process each image
            all_text = []
            for i, image in enumerate(images, 1):
                image_path = os.path.join(temp_dir, f"slide_{i}.png")
                image.save(image_path, 'PNG')

                # Convert image to base64
                with open(image_path, "rb") as img_file:
                    base64_encoded = base64.b64encode(
                        img_file.read()).decode('utf-8')
                    data_uri = f"data:image/png;base64,{base64_encoded}"

                # Custom prompt for slide content extraction
                prompt = """
                Extract all text content from this PowerPoint slide image. 
                Pay attention to:
                1. Slide titles and headers
                2. Bullet points and text content
                3. Text in diagrams and charts
                4. Table contents (preserve structure)
                
                Describe any visual elements like charts, diagrams, or images if they contain important information.
                """

                # Extract text using the OpenAI vision API
                try:
                    extracted_text = self.openai_helper.get_openai_response_image(
                        data_uri, prompt=prompt
                    )

                    all_text.append({
                        "text": extracted_text,
                        "page_number": i
                    })
                    print(f"   ✓ Processed slide {i}/{len(images)}")
                except Exception as e:
                    print(
                        f"   ⚠️ Error processing image for slide {i}: {str(e)}")
                    all_text.append({
                        "text": f"[Error processing slide {i}: {str(e)}]",
                        "page_number": i
                    })

            # Log success
            self.log_success(file_path)

            return all_text

        except Exception as e:
            error_msg = f"Error parsing PPT file with PDF bridge parser: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.log_failure(file_path, str(e), "PDF_BRIDGE_PARSING_ERROR")
            raise

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("   ✓ Cleaned up temporary files")

    def _convert_with_comtypes(self, ppt_path, pdf_path):
        """Convert PPTX to PDF using comtypes (Windows-only)"""
        # Check if running on Windows
        if os.name != 'nt':
            raise ValueError("comtypes conversion only works on Windows")

        print("   • Converting using comtypes (Windows COM)...")
        import comtypes.client

        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        powerpoint.Visible = True

        try:
            deck = powerpoint.Presentations.Open(os.path.abspath(ppt_path))
            # 32 is the PDF format code
            deck.SaveAs(os.path.abspath(pdf_path), 32)
            deck.Close()
        finally:
            powerpoint.Quit()

    def _convert_with_libreoffice(self, ppt_path, pdf_path):
        """Convert PPTX to PDF using LibreOffice"""
        print("   • Converting using LibreOffice...")

        # Check for LibreOffice/soffice in common locations
        soffice_paths = [
            "soffice",  # if in PATH
            "/usr/bin/soffice",
            "/usr/lib/libreoffice/program/soffice",
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
            "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]

        soffice_path = None
        for path in soffice_paths:
            try:
                # Use 'which' on Unix or 'where' on Windows to check if command exists
                if os.name == 'nt':  # Windows
                    check_cmd = f"where {path}"
                else:  # Unix-like
                    check_cmd = f"which {path}"

                result = os.system(check_cmd + " > " + os.devnull + " 2>&1")
                if result == 0 or os.path.isfile(path):
                    soffice_path = path
                    break
            except:
                continue

        if not soffice_path:
            raise ValueError("LibreOffice not found")

        # Convert using LibreOffice headless mode
        output_dir = os.path.dirname(pdf_path)
        cmd = f'"{soffice_path}" --headless --convert-to pdf --outdir "{output_dir}" "{ppt_path}"'

        result = os.system(cmd)

        if result != 0:
            raise ValueError(
                f"LibreOffice conversion failed with exit code {result}")

        # Find the output PDF and rename if needed
        base_name = os.path.splitext(os.path.basename(ppt_path))[0]
        generated_pdf = os.path.join(output_dir, f"{base_name}.pdf")

        if os.path.exists(generated_pdf) and generated_pdf != pdf_path:
            os.rename(generated_pdf, pdf_path)

    def _convert_with_unoconv(self, ppt_path, pdf_path):
        """Convert PPTX to PDF using unoconv"""
        print("   • Converting using unoconv...")

        # Check if unoconv is available
        result = os.system("unoconv --version > " + os.devnull + " 2>&1")
        if result != 0:
            raise ValueError(
                "unoconv not found. Install with 'pip install unoconv' or system package manager")

        # Convert using unoconv
        cmd = f'unoconv -f pdf -o "{pdf_path}" "{ppt_path}"'
        result = os.system(cmd)

        if result != 0:
            raise ValueError(
                f"unoconv conversion failed with exit code {result}")
