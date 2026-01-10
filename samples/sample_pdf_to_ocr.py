#!/usr/bin/env python3
"""
Sample: PDF to OCR (plain text extraction)

Minimal example showing how to extract plain text from PDFs using the DeepSeek-OCR library.
"""

from pathlib import Path
from Lib import get_config, OCRClient, PostProcessor
import time


def extract_text(pdf_path: str, output_dir: str = "data/results"):
    """Extract plain text from a PDF."""
    
    # Initialize components
    config = get_config()
    client = OCRClient(config)
    postprocessor = PostProcessor(config)
    
    # Check API health
    healthy, _, _ = client.check_health()
    if not healthy:
        print("? API not available")
        return None
    
    print(f"?? Extracting text from: {pdf_path}")
    
    # Create job with OCR prompt
    job_id, error = client.create_job(pdf_path, "<image>Extract all text from the image.")
    if error:
        print(f"? Failed: {error}")
        return None
    
    # Wait for completion
    while True:
        status, _ = client.get_job_status(job_id)
        if status and status['status'] == 'completed':
            break
        elif status and status['status'] == 'failed':
            print(f"? Failed: {status.get('error')}")
            return None
        time.sleep(2)
    
    # Download and clean result
    result, _ = client.download_result(job_id)
    cleaned, _, _ = postprocessor.process(result, remove_page_splits=True)
    
    # Save output
    output_path = Path(output_dir) / f"{Path(pdf_path).stem}_OCR.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding='utf-8')
    
    print(f"? Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sample_pdf_to_ocr.py <pdf_path>")
        sys.exit(1)
    
    extract_text(sys.argv[1])
