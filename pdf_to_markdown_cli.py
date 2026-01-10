#!/usr/bin/env python3
"""
DeepSeek-OCR Command Line Interface

A comprehensive CLI tool for converting PDF files to Markdown using the DeepSeek-OCR API.

Features:
- Single PDF or batch processing
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Post-processing options (tag cleanup, image extraction, page split removal)
- Progress display
- Input from file list
- Configurable output directory

Usage:
    python pdf_to_markdown_cli.py input.pdf
    python pdf_to_markdown_cli.py input.pdf -o output/
    python pdf_to_markdown_cli.py folder/ --batch
    python pdf_to_markdown_cli.py --list files.txt
    python pdf_to_markdown_cli.py input.pdf --mode ocr
    python pdf_to_markdown_cli.py input.pdf --prompt custom_prompt.yaml
    python pdf_to_markdown_cli.py input.pdf --extract-images --no-page-splits
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from Lib import Config, config, OCRClient, PostProcessor, FileManager


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class PDFConverter:
    """PDF to Markdown converter using DeepSeek-OCR API."""
    
    # Predefined prompts for different modes
    PROMPTS = {
        'markdown': '<image>Convert the content of the image to markdown.',
        'ocr': '<image>Extract all text from the image.',
    }
    
    def __init__(
        self,
        api_host: str = 'localhost',
        api_port: int = 8000,
        output_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the converter.
        
        Args:
            api_host: API server host
            api_port: API server port
            output_dir: Output directory (default: same as input)
            verbose: Enable verbose logging
        """
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize config
        self.config = Config(
            api_host=api_host,
            api_port=api_port,
        )
        if output_dir:
            self.config.results_dir = Path(output_dir)
        self.config.ensure_directories()
        
        # Initialize components
        self.client = OCRClient(self.config)
        self.postprocessor = PostProcessor(self.config)
        self.file_manager = FileManager(self.config)
        
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
    
    def check_api(self) -> bool:
        """Check if API is available."""
        healthy, msg, available = self.client.check_health()
        if not healthy:
            print(f"{Colors.RED}? API not available: {msg}{Colors.RESET}")
            return False
        if self.verbose:
            print(f"{Colors.GREEN}? API connected{Colors.RESET}")
        return True
    
    def wait_for_availability(self, timeout: int = 120) -> bool:
        """Wait for server to become available."""
        checks = timeout // self.config.poll_interval
        for _ in range(checks):
            available, _ = self.client.check_available()
            if available:
                return True
            time.sleep(self.config.poll_interval)
        return False
    
    def convert_pdf(
        self,
        pdf_path: str,
        prompt: str,
        extract_images: bool = False,
        remove_page_splits: bool = True,
        raw_output: bool = False,
        suffix: str = '_MD'
    ) -> Optional[str]:
        """
        Convert a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            prompt: Processing prompt
            extract_images: Extract images from detected regions
            remove_page_splits: Remove page split markers
            raw_output: Skip post-processing (for custom prompts)
            suffix: Output filename suffix
            
        Returns:
            Path to output file, or None on error
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"{Colors.RED}? File not found: {pdf_path}{Colors.RESET}")
            return None
        
        # Determine output path
        if self.output_dir:
            output_path = self.output_dir / f"{pdf_path.stem}{suffix}.md"
        else:
            output_path = pdf_path.with_name(f"{pdf_path.stem}{suffix}.md")
        
        print(f"{Colors.CYAN}?? Processing: {pdf_path.name}{Colors.RESET}")
        
        # Wait for server availability
        if not self.wait_for_availability():
            print(f"{Colors.RED}? Server busy, timeout waiting{Colors.RESET}")
            return None
        
        # Create job
        job_id, error = self.client.create_job(str(pdf_path), prompt)
        if error:
            print(f"{Colors.RED}? Job creation failed: {error}{Colors.RESET}")
            return None
        
        if self.verbose:
            print(f"   Job ID: {job_id}")
        
        # Poll for completion with progress display
        last_progress = -1
        while True:
            status, error = self.client.get_job_status(job_id)
            if error:
                print(f"{Colors.RED}? Status error: {error}{Colors.RESET}")
                return None
            
            job_status = status.get('status')
            progress = status.get('progress', 0)
            
            # Show progress
            if progress != last_progress:
                processed = status.get('processed_pages', 0)
                total = status.get('total_pages', 0)
                print(f"   ? {progress:.0f}% ({processed}/{total} pages)", end='\r')
                last_progress = progress
            
            if job_status == 'completed':
                print()  # New line after progress
                break
            elif job_status == 'failed':
                print(f"\n{Colors.RED}? Job failed: {status.get('error')}{Colors.RESET}")
                return None
            
            time.sleep(self.config.poll_interval)
        
        # Download result
        result, error = self.client.download_result(job_id)
        if error:
            print(f"{Colors.RED}? Download failed: {error}{Colors.RESET}")
            return None
        
        # Post-process (unless raw output requested)
        if raw_output:
            output_content = result
            image_paths = []
            zip_path = None
        else:
            output_content, image_paths, zip_path = self.postprocessor.process(
                result,
                pdf_path=str(pdf_path),
                extract_images=extract_images,
                remove_page_splits=remove_page_splits,
                create_zip=extract_images,
                pdf_filename=pdf_path.name
            )
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding='utf-8')
        
        print(f"{Colors.GREEN}   ? Saved: {output_path}{Colors.RESET}")
        
        if image_paths:
            print(f"{Colors.BLUE}   ???  Extracted {len(image_paths)} images{Colors.RESET}")
        if zip_path:
            print(f"{Colors.BLUE}   ?? Images ZIP: {zip_path}{Colors.RESET}")
        
        return str(output_path)
    
    def process_batch(
        self,
        pdf_paths: List[str],
        prompt: str,
        extract_images: bool = False,
        remove_page_splits: bool = True,
        raw_output: bool = False,
        suffix: str = '_MD'
    ) -> List[str]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            prompt: Processing prompt
            extract_images: Extract images from detected regions
            remove_page_splits: Remove page split markers
            raw_output: Skip post-processing
            suffix: Output filename suffix
            
        Returns:
            List of output file paths
        """
        total = len(pdf_paths)
        results = []
        failed = []
        
        print(f"{Colors.BOLD}?? Processing {total} PDF files{Colors.RESET}\n")
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"{Colors.BOLD}[{i}/{total}]{Colors.RESET}", end=' ')
            
            output = self.convert_pdf(
                pdf_path,
                prompt,
                extract_images=extract_images,
                remove_page_splits=remove_page_splits,
                raw_output=raw_output,
                suffix=suffix
            )
            
            if output:
                results.append(output)
            else:
                failed.append(pdf_path)
            
            print()  # Spacing between files
        
        # Summary
        print(f"\n{Colors.BOLD}{'='*50}{Colors.RESET}")
        print(f"{Colors.GREEN}? Successful: {len(results)}{Colors.RESET}")
        if failed:
            print(f"{Colors.RED}? Failed: {len(failed)}{Colors.RESET}")
            for f in failed:
                print(f"   - {f}")
        
        return results


def get_pdf_files_from_list(list_file: str) -> List[str]:
    """Read PDF file paths from a text file (one per line)."""
    paths = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if Path(line).exists():
                    paths.append(line)
                else:
                    print(f"{Colors.YELLOW}??  File not found: {line}{Colors.RESET}")
    return paths


def get_pdf_files_from_folder(folder: str) -> List[str]:
    """Get all PDF files from a folder."""
    return [str(p) for p in Path(folder).glob("*.pdf")]


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF files to Markdown using DeepSeek-OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pdf                     # Convert single PDF
  %(prog)s input.pdf -o output/          # Specify output directory
  %(prog)s folder/ --batch               # Process all PDFs in folder
  %(prog)s --list files.txt              # Process files listed in text file
  %(prog)s input.pdf --mode ocr          # Extract plain text
  %(prog)s input.pdf --prompt custom.yaml  # Use custom prompt
  %(prog)s input.pdf --extract-images    # Extract images to ZIP
  %(prog)s input.pdf --no-clean          # Raw output (no post-processing)
        """
    )
    
    # Input options
    parser.add_argument('input', nargs='?', help='PDF file or folder path')
    parser.add_argument('--list', '-l', metavar='FILE', help='File containing list of PDFs (one per line)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all PDFs in input folder')
    
    # Output options
    parser.add_argument('--output', '-o', metavar='DIR', help='Output directory (default: same as input)')
    parser.add_argument('--suffix', '-s', default='_MD', help='Output filename suffix (default: _MD)')
    
    # Processing mode
    parser.add_argument('--mode', '-m', choices=['markdown', 'ocr'], default='markdown',
                        help='Processing mode (default: markdown)')
    parser.add_argument('--prompt', '-p', metavar='FILE', help='Custom prompt YAML file')
    
    # Post-processing options
    parser.add_argument('--extract-images', '-i', action='store_true', help='Extract images to ZIP')
    parser.add_argument('--keep-page-splits', action='store_true', help='Keep page split markers')
    parser.add_argument('--no-clean', action='store_true', help='Skip post-processing (raw output)')
    
    # Connection options
    parser.add_argument('--host', default='localhost', help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='API port (default: 8000)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine input files
    pdf_files = []
    
    if args.list:
        pdf_files = get_pdf_files_from_list(args.list)
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_dir() or args.batch:
            pdf_files = get_pdf_files_from_folder(args.input)
        elif input_path.is_file():
            pdf_files = [args.input]
        else:
            print(f"{Colors.RED}? Input not found: {args.input}{Colors.RESET}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    if not pdf_files:
        print(f"{Colors.YELLOW}No PDF files to process{Colors.RESET}")
        sys.exit(0)
    
    # Determine prompt
    if args.prompt:
        prompt = FileManager.load_custom_prompt(args.prompt)
        if not prompt:
            print(f"{Colors.RED}? Could not load prompt from {args.prompt}{Colors.RESET}")
            sys.exit(1)
        suffix = args.suffix if args.suffix != '_MD' else '_CUSTOM'
        raw_output = True  # Don't clean custom prompt output by default
    else:
        prompt = PDFConverter.PROMPTS[args.mode]
        suffix = args.suffix if args.suffix != '_MD' else ('_MD' if args.mode == 'markdown' else '_OCR')
        raw_output = args.no_clean
    
    # Initialize converter
    try:
        converter = PDFConverter(
            api_host=args.host,
            api_port=args.port,
            output_dir=args.output,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"{Colors.RED}? Initialization error: {e}{Colors.RESET}")
        sys.exit(1)
    
    # Check API connection
    if not converter.check_api():
        sys.exit(1)
    
    # Process files
    if len(pdf_files) == 1:
        result = converter.convert_pdf(
            pdf_files[0],
            prompt,
            extract_images=args.extract_images,
            remove_page_splits=not args.keep_page_splits,
            raw_output=raw_output,
            suffix=suffix
        )
        sys.exit(0 if result else 1)
    else:
        results = converter.process_batch(
            pdf_files,
            prompt,
            extract_images=args.extract_images,
            remove_page_splits=not args.keep_page_splits,
            raw_output=raw_output,
            suffix=suffix
        )
        sys.exit(0 if results else 1)


if __name__ == '__main__':
    main()
