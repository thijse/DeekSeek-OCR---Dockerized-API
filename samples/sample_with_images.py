#!/usr/bin/env python3
"""
Sample: PDF conversion with image extraction

Minimal example showing how to extract images from PDFs during conversion.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Lib import config, OCRClient, PostProcessor


def convert_with_images(pdf_path: str, output_dir: str = None):
    """Convert PDF to Markdown and extract images."""
    
    output_dir = output_dir or str(config.results_dir)
    
    # Initialize components
    client = OCRClient(config)
    postprocessor = PostProcessor(config)
    
    print(f"?? Processing: {pdf_path}")
    
    # Create job
    job_id, error = client.create_job(pdf_path, "<image>Convert the content of the image to markdown.")
    if error:
        print(f"? Failed: {error}")
        return None
    
    # Wait for completion
    while True:
        status, _ = client.get_job_status(job_id)
        if status and status['status'] == 'completed':
            break
        elif status and status['status'] == 'failed':
            return None
        time.sleep(2)
    
    # Download result
    result, _ = client.download_result(job_id)
    
    # Post-process with image extraction
    cleaned, image_paths, zip_path = postprocessor.process(
        result,
        pdf_path=pdf_path,
        extract_images=True,
        remove_page_splits=True,
        create_zip=True,
        pdf_filename=Path(pdf_path).name
    )
    
    # Save markdown
    output_path = Path(output_dir) / f"{Path(pdf_path).stem}_MD.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding='utf-8')
    
    print(f"? Markdown: {output_path}")
    if image_paths:
        print(f"???  Images: {len(image_paths)} extracted")
    if zip_path:
        print(f"?? ZIP: {zip_path}")
    
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sample_with_images.py <pdf_path>")
        sys.exit(1)
    
    convert_with_images(sys.argv[1])
