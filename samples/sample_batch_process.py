#!/usr/bin/env python3
"""
Sample: Batch PDF processing

Minimal example showing how to process multiple PDFs from a folder.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Lib import config, OCRClient, PostProcessor


def process_folder(folder_path: str, output_dir: str = None):
    """Process all PDFs in a folder."""
    
    output_dir = output_dir or str(config.results_dir)
    
    # Initialize components
    client = OCRClient(config)
    postprocessor = PostProcessor(config)
    
    # Find all PDFs
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
    
    print(f"?? Found {len(pdf_files)} PDF files")
    
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        # Wait for server availability
        while True:
            available, _ = client.check_available()
            if available:
                break
            time.sleep(2)
        
        # Create job
        job_id, error = client.create_job(str(pdf_path), "<image>Convert the content of the image to markdown.")
        if error:
            print(f"  ? Failed: {error}")
            continue
        
        # Wait for completion
        while True:
            status, _ = client.get_job_status(job_id)
            if status and status['status'] == 'completed':
                break
            elif status and status['status'] == 'failed':
                print(f"  ? Failed: {status.get('error')}")
                break
            time.sleep(2)
        else:
            continue
        
        # Download and save
        result, _ = client.download_result(job_id)
        cleaned, _, _ = postprocessor.process(result, remove_page_splits=True)
        
        output_path = Path(output_dir) / f"{pdf_path.stem}_MD.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding='utf-8')
        
        print(f"  ? Saved: {output_path}")
        results.append(str(output_path))
    
    print(f"\n?? Completed: {len(results)}/{len(pdf_files)} files")
    return results


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    process_folder(folder)
