#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface

A web interface for converting PDF files to Markdown or OCR text using the DeepSeek OCR API.
Supports multiple processing modes:
- Markdown conversion (structured document)
- OCR extraction (plain text)
- Custom prompt processing (user-defined YAML prompt)
"""

import gradio as gr
import requests
import base64
import tempfile
import os
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepSeekOCRClient:
    """Client for interacting with DeepSeek OCR API"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.health_endpoint = f"{api_base_url}/health"
        self.pdf_endpoint = f"{api_base_url}/ocr/pdf"
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return True, f"API is healthy. Model: {data.get('model_name', 'Unknown')}"
            else:
                return False, f"API returned status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to API: {str(e)}"
    
    def process_pdf(self, pdf_path, prompt, progress=None):
        """
        Process a PDF file with the specified prompt
        
        Args:
            pdf_path: Path to the PDF file
            prompt: The prompt to use for processing
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (success, result_text)
        """
        try:
            # Update progress
            if progress:
                progress(0.1, desc="Reading PDF file...")
            
            # Read the PDF file
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            if progress:
                progress(0.3, desc="Uploading to API...")
            
            # Prepare the multipart form data
            files = {
                'file': ('document.pdf', pdf_data, 'application/pdf')
            }
            data = {
                'prompt': prompt
            }
            
            if progress:
                progress(0.5, desc="Processing PDF...")
            
            # Make the API request
            response = requests.post(
                self.pdf_endpoint,
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
            
            if progress:
                progress(0.9, desc="Formatting results...")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    markdown_text = result.get('result', '')
                    page_count = result.get('page_count', 0)
                    
                    if progress:
                        progress(1.0, desc="Complete!")
                    
                    return True, f"✅ Successfully processed {page_count} page(s)\n\n{markdown_text}"
                else:
                    error = result.get('error', 'Unknown error')
                    return False, f"❌ Processing failed: {error}"
            else:
                return False, f"❌ API error: Status code {response.status_code}\n{response.text}"
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return False, f"❌ Error: {str(e)}"


# Initialize the client
client = DeepSeekOCRClient()


def load_custom_prompt():
    """Load custom prompt from YAML file"""
    try:
        custom_prompt_file = "custom_prompt.yaml"
        if os.path.exists(custom_prompt_file):
            with open(custom_prompt_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('prompt', '<image>\nFree OCR.')
        return '<image>\nFree OCR.'
    except Exception as e:
        logger.warning(f"Could not load custom prompt: {e}")
        return '<image>\nFree OCR.'


def process_pdf_interface(pdf_file, processing_mode, custom_prompt_text, progress=gr.Progress()):
    """
    Process PDF file based on selected mode
    
    Args:
        pdf_file: Uploaded PDF file
        processing_mode: One of "Markdown", "OCR", or "Custom Prompt"
        custom_prompt_text: Custom prompt text (only used in Custom Prompt mode)
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status_message, result_text)
    """
    if pdf_file is None:
        return "⚠️ Please upload a PDF file", ""
    
    # Check API health first
    is_healthy, health_msg = client.check_health()
    if not is_healthy:
        return f"⚠️ API Error: {health_msg}\n\nPlease ensure the DeepSeek-OCR API is running at http://localhost:8000", ""
    
    # Determine the prompt based on processing mode
    if processing_mode == "Markdown":
        prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    elif processing_mode == "OCR":
        prompt = '<image>\nFree OCR.'
    elif processing_mode == "Custom Prompt":
        if not custom_prompt_text.strip():
            return "⚠️ Please enter a custom prompt", ""
        prompt = custom_prompt_text
    else:
        return "⚠️ Invalid processing mode", ""
    
    # Process the PDF
    success, result = client.process_pdf(pdf_file.name, prompt, progress)
    
    if success:
        status = f"✅ Processing complete!\n\nMode: {processing_mode}\nPrompt: {prompt}"
        return status, result
    else:
        return result, ""


def check_api_status():
    """Check and return API status"""
    is_healthy, msg = client.check_health()
    if is_healthy:
        return f"✅ {msg}"
    else:
        return f"❌ {msg}"


def update_prompt_visibility(processing_mode):
    """Show/hide custom prompt textbox based on processing mode"""
    is_custom = (processing_mode == "Custom Prompt")
    return gr.update(visible=is_custom)


# Create the Gradio interface
with gr.Blocks(title="DeepSeek-OCR PDF Converter", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📄 DeepSeek-OCR PDF Converter
    
    Convert PDF documents to Markdown or extract text using DeepSeek-OCR API.
    
    **Supported Modes:**
    - **Markdown**: Preserves document structure (headings, paragraphs, formatting)
    - **OCR**: Extracts plain text without formatting
    - **Custom Prompt**: Use your own processing prompt
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # API Status
            api_status = gr.Textbox(
                label="API Status",
                value=check_api_status(),
                interactive=False,
                lines=2
            )
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            gr.Markdown("---")
            
            # File upload
            pdf_input = gr.File(
                label="Upload PDF File",
                file_types=[".pdf"],
                type="filepath"
            )
            
            # Processing mode
            processing_mode = gr.Radio(
                choices=["Markdown", "OCR", "Custom Prompt"],
                value="Markdown",
                label="Processing Mode",
                info="Select how to process the PDF"
            )
            
            # Custom prompt input (initially hidden)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt",
                placeholder="Enter your custom prompt here...\nExample: <image>\nExtract all tables.",
                lines=3,
                visible=False,
                info="Loaded from custom_prompt.yaml or enter your own"
            )
            
            # Load custom prompt button
            load_prompt_btn = gr.Button("📂 Load from custom_prompt.yaml", size="sm", visible=False)
            
            # Process button
            process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # Status message
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )
            
            # Result output
            result_output = gr.Textbox(
                label="Result",
                interactive=False,
                lines=20,
                max_lines=50,
                show_copy_button=True
            )
    
    gr.Markdown("""
    ---
    ### 📝 Instructions
    
    1. **Start the API**: Ensure the DeepSeek-OCR API is running on `http://localhost:8000`
       - Using Docker: `docker-compose up -d`
       - Check status with the Refresh button above
    
    2. **Upload PDF**: Click the upload area and select a PDF file
    
    3. **Choose Mode**: Select your preferred processing mode
       - **Markdown**: Best for documents with structure
       - **OCR**: Best for simple text extraction
       - **Custom Prompt**: Use your own processing instructions
    
    4. **Process**: Click "Process PDF" and wait for results
    
    5. **Copy Results**: Use the copy button to save the output
    
    ### 🔧 Tips
    
    - For large PDFs, processing may take several minutes
    - The API must be running before processing
    - Custom prompts should start with `<image>` tag
    - Results can be saved directly from the text box
    """)
    
    # Event handlers
    refresh_btn.click(
        fn=check_api_status,
        outputs=api_status
    )
    
    processing_mode.change(
        fn=update_prompt_visibility,
        inputs=processing_mode,
        outputs=custom_prompt_input
    ).then(
        fn=lambda mode: gr.update(visible=(mode == "Custom Prompt")),
        inputs=processing_mode,
        outputs=load_prompt_btn
    )
    
    load_prompt_btn.click(
        fn=load_custom_prompt,
        outputs=custom_prompt_input
    )
    
    process_btn.click(
        fn=process_pdf_interface,
        inputs=[pdf_input, processing_mode, custom_prompt_input],
        outputs=[status_output, result_output]
    )


# Launch the app
if __name__ == "__main__":
    # Check API on startup
    is_healthy, msg = client.check_health()
    if is_healthy:
        logger.info(f"✅ API is ready: {msg}")
    else:
        logger.warning(f"⚠️ API not available: {msg}")
        logger.warning("Please start the DeepSeek-OCR API: docker-compose up -d")
    
    # Launch Gradio
    app.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7861,  # Using 7861 to avoid port conflicts
        share=False,  # Don't create public link by default
        show_error=True,
        inbrowser=True  # Automatically open in browser
    )