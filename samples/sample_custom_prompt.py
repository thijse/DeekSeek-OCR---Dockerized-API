#!/usr/bin/env python3
"""
Sample: PDF conversion with custom prompt

Minimal example showing how to use a custom prompt from YAML file.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Lib import config, OCRClient, PostProcessor, FileManager


def convert_with_custom_prompt(pdf_path: str, prompt_file: str = "custom_prompt.yaml"):
    """Convert PDF using a custom prompt from YAML file."""
    
    # Initialize components
    client = OCRClient(config)
    postprocessor = PostProcessor(config)
    
    # Load custom prompt
    prompt = FileManager.load_custom_prompt(prompt_file)
    if not prompt:
        print(f"? Could not load prompt from {prompt_file}")
        return None
    
    print(f"?? Processing: {pdf_path}")
    print(f"?? Prompt: {prompt[:50]}...")
    
    # Create job with custom prompt
    job_id, error = client.create_job(pdf_path, prompt)
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
        print(f"   Progress: {status['progress']:.0f}%")
        time.sleep(2)
    
    # Download and save result (raw, no post-processing for custom prompts)
    result, _ = client.download_result(job_id)
    
    output_path = config.results_dir / f"{Path(pdf_path).stem}_CUSTOM.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding='utf-8')
    
    print(f"? Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sample_custom_prompt.py <pdf_path> [prompt_file.yaml]")
        sys.exit(1)
    
    pdf = sys.argv[1]
    prompt_file = sys.argv[2] if len(sys.argv) > 2 else "custom_prompt.yaml"
    convert_with_custom_prompt(pdf, prompt_file)
