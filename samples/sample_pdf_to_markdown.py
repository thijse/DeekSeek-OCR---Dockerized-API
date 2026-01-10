#!/usr/bin/env python3
"""
Sample: PDF to Markdown conversion

Minimal example showing how to convert PDFs to Markdown using the DeepSeek-OCR library.
"""

from pathlib import Path
from Lib import get_config, OCRClient, PostProcessor, FileManager
import time


def convert_pdf(pdf_path: str, output_dir: str = "data/results"):
    """Convert a single PDF to Markdown."""
    
    # Initialize components
    config = get_config()
    client = OCRClient(config)
    postprocessor = PostProcessor(config)
    
    # Check API health
    healthy, msg, available = client.check_health()
    if not healthy:
        print(f"? API not available: {msg}")
        return None
    
    print(f"?? Processing: {pdf_path}")
    
    # Create job
    job_id, error = client.create_job(pdf_path, "<image>Convert the content of the image to markdown.")
    if error:
        print(f"? Failed to create job: {error}")
        return None
    
    print(f"? Job created: {job_id}")
    
    # Poll for completion
    while True:
        status, error = client.get_job_status(job_id)
        if error:
            print(f"? Status error: {error}")
            return None
        
        if status['status'] == 'completed':
            break
        elif status['status'] == 'failed':
            print(f"? Job failed: {status.get('error')}")
            return None
        
        print(f"   Progress: {status['progress']:.0f}%")
        time.sleep(2)
    
    # Download result
    result, error = client.download_result(job_id)
    if error:
        print(f"? Download error: {error}")
        return None
    
    # Post-process (clean tags, remove page splits)
    cleaned, _, _ = postprocessor.process(result, remove_page_splits=True)
    
    # Save output
    output_path = Path(output_dir) / f"{Path(pdf_path).stem}_MD.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding='utf-8')
    
    print(f"? Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sample_pdf_to_markdown.py <pdf_path>")
        sys.exit(1)
    
    convert_pdf(sys.argv[1])
