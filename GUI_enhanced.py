#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface (Enhanced)

A web interface for converting PDF files to Markdown or OCR text using the DeepSeek OCR API.
Features:
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Extended timeout for large documents (up to 2 hours)
- File persistence - uploads saved to data/uploads folder
- Result persistence - results auto-saved to data/results folder
- Queue system - add files one by one, process sequentially
- Real-time progress updates
"""

import gradio as gr
import requests
import os
import re
import io
import yaml
import json
import shutil
import hashlib
import urllib.parse
import zipfile
import fitz  # PyMuPDF for page count
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging
import time
import threading
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
IMAGES_DIR = Path("data/images")
TIMEOUT_SECONDS = 7200  # 2 hours - enough for 500+ pages at ~15s/page
ESTIMATED_SECONDS_PER_PAGE = 15  # Initial estimate, updated during processing

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Global state for adaptive timing
processing_stats = {
    "total_pages_processed": 0,
    "total_time_spent": 0,
    "avg_seconds_per_page": ESTIMATED_SECONDS_PER_PAGE
}

# Global cancel flag for stopping processing
cancel_requested = False

# Global queue state
file_queue = []
processing_status = {"current": "", "progress": 0, "is_processing": False}


class DeepSeekOCRClient:
    """Client for interacting with DeepSeek OCR API"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.health_endpoint = f"{api_base_url}/health"
        self.pdf_endpoint = f"{api_base_url}/ocr/pdf"
        self.current_job = None
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return True, f"✅ API is healthy. Model: {data.get('model_name', 'DeepSeek-OCR')}"
            else:
                return False, f"❌ API returned status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Cannot connect to API: {str(e)}"
    
    def get_pdf_page_count(self, pdf_path):
        """Get the number of pages in a PDF"""
        try:
            doc = fitz.open(pdf_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            return None
    
    def process_pdf(self, pdf_path, prompt, timeout=TIMEOUT_SECONDS):
        """
        Process a PDF file with the specified prompt.
        Handles both single-result and per-page batch responses from the API.
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()

            files = {'file': (os.path.basename(pdf_path), pdf_data, 'application/pdf')}
            data = {'prompt': prompt}

            logger.info(f"Starting PDF processing with {timeout}s timeout...")

            response = requests.post(
                self.pdf_endpoint,
                files=files,
                data=data,
                timeout=timeout
            )

            if response.status_code != 200:
                return False, f"API error: Status code {response.status_code}\n{response.text}", 0

            result = response.json()
            if not result.get('success', False):
                error = result.get('error', 'Unknown error')
                return False, f"Processing failed: {error}", 0

            # Handle batch response with per-page results
            markdown_text = ""
            page_count = result.get('page_count', 0)
            if isinstance(result, dict) and isinstance(result.get('results'), list):
                pages = []
                for page in result['results']:
                    if isinstance(page, dict):
                        page_text = page.get('result') or page.get('markdown') or page.get('content') or page.get('text')
                        if page_text:
                            pages.append(page_text)
                if pages:
                    markdown_text = "\n\n<--- Page Split --->\n\n".join(pages)
                    page_count = page_count or len(pages)

            # Fall back to single-field result
            if not markdown_text:
                markdown_text = (
                    result.get('result')
                    or result.get('markdown')
                    or result.get('content')
                    or result.get('text')
                    or ""
                )

            if not markdown_text:
                return False, "Processing finished but returned empty content.", page_count

            return True, markdown_text, page_count

        except requests.exceptions.Timeout:
            return False, f"Request timed out after {timeout} seconds. The document may be too large.", 0
        except requests.exceptions.ConnectionError:
            return False, "Connection error. Please check if the API is running.", 0
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return False, f"Error: {str(e)}", 0


# Initialize the client
client = DeepSeekOCRClient()


# =============================================================================
# Post-Processing Functions (from pdf_to_markdown_processor_enhanced.py)
# =============================================================================

def re_match_tags(text: str) -> Tuple[List, List, List]:
    """
    Match reference patterns in the text
    
    Args:
        text: The text to search for patterns
        
    Returns:
        Tuple of (all_matches, image_matches, other_matches)
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    
    return matches, matches_image, matches_other


def pdf_to_images(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion
        
    Returns:
        List of PIL Images
    """
    images = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
    
    return images


def extract_and_save_images(pdf_path: str, content: str, page_idx: int = 0) -> Tuple[str, int, List[str]]:
    """
    Extract images from content based on coordinate tags and save them.
    
    Args:
        pdf_path: Path to the original PDF file
        content: The OCR content with reference tags
        page_idx: Index of the page being processed (default 0 for single-page processing)
        
    Returns:
        Tuple of (processed_content, number_of_images_extracted, list_of_image_paths)
    """
    extracted_paths = []
    
    # Get PDF images
    logger.info(f"Attempting to extract images from: {pdf_path}")
    pdf_images = pdf_to_images(pdf_path)
    if not pdf_images:
        logger.warning(f"No images could be extracted from PDF: {pdf_path}")
        # If we can't extract images, just clean up the tags
        _, matches_images, _ = re_match_tags(content)
        for tag in matches_images:
            content = content.replace(tag, '[Image]', 1)
        return content, 0, []
    
    logger.info(f"Loaded {len(pdf_images)} pages from PDF")
    
    # Find all image references
    _, matches_images, _ = re_match_tags(content)
    logger.info(f"Found {len(matches_images)} image tags in content")
    total_extracted = 0
    
    # For multi-page batch results, we'll need to cycle through pages
    # Since OCR output doesn't include page markers, we'll distribute images across pages
    current_page_idx = 0
    
    # Process each image tag
    for img_idx, img_tag in enumerate(matches_images):
        try:
            # Extract the reference text
            pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
            det_match = re.search(pattern, img_tag)
            
            if det_match:
                det_content = det_match.group(1)
                logger.debug(f"Found image coordinates: {det_content[:100]}...")
                try:
                    coordinates = eval(det_content)
                    
                    # Try to use appropriate page (cycle through pages for multi-page docs)
                    if len(pdf_images) > 1:
                        # For multi-page, use modulo to cycle through pages
                        # This is a heuristic - ideally the OCR output would include page markers
                        page_to_use = img_idx % len(pdf_images)
                    else:
                        page_to_use = 0
                    
                    page_image = pdf_images[page_to_use]
                    image_width, image_height = page_image.size
                    
                    for points in coordinates:
                        x1, y1, x2, y2 = points
                        
                        # Scale coordinates to actual image size (coords are 0-999)
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        logger.debug(f"Cropping region: ({x1}, {y1}, {x2}, {y2}) from page {page_to_use}")
                        
                        # Ensure valid crop coordinates
                        if x1 >= x2 or y1 >= y2:
                            logger.warning(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
                            continue
                        
                        # Crop and save the image
                        cropped = page_image.crop((x1, y1, x2, y2))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_filename = f"{Path(pdf_path).stem}_img{total_extracted}_{timestamp}.jpg"
                        image_path = IMAGES_DIR / image_filename
                        cropped.save(image_path)
                        extracted_paths.append(str(image_path))
                        logger.info(f"Saved image: {image_path}")
                        
                        # Replace reference with markdown link
                        encoded_filename = urllib.parse.quote(image_filename)
                        markdown_link = f"\n![Extracted Image](images/{encoded_filename})\n"
                        content = content.replace(img_tag, markdown_link, 1)
                        
                        total_extracted += 1
                        break  # Only extract first bounding box per tag
                except Exception as e:
                    logger.error(f"Error processing image coordinates: {str(e)}")
                    content = content.replace(img_tag, '[Image - extraction failed]', 1)
        except Exception as e:
            logger.error(f"Error extracting image: {str(e)}")
            content = content.replace(img_tag, '[Image - error]', 1)
    
    logger.info(f"Total images extracted: {total_extracted}")
    return content, total_extracted, extracted_paths


def create_images_zip(image_paths: List[str], pdf_filename: str) -> Optional[str]:
    """
    Create a zip file containing extracted images.
    
    Args:
        image_paths: List of paths to extracted images
        pdf_filename: Original PDF filename (used for zip name)
        
    Returns:
        Path to the created zip file, or None if no images
    """
    if not image_paths:
        return None
    
    # Create zip filename based on PDF name
    base_name = Path(pdf_filename).stem
    zip_filename = f"{base_name}_images.zip"
    zip_path = RESULTS_DIR / zip_filename
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_path in image_paths:
                # Add image with just the filename (no folder structure)
                zipf.write(img_path, Path(img_path).name)
        
        logger.info(f"Created images zip: {zip_path} with {len(image_paths)} images")
        return str(zip_path)
    except Exception as e:
        logger.error(f"Error creating images zip: {str(e)}")
        return None


def clean_content(content: str, extract_images: bool = False, pdf_path: str = None, remove_page_splits: bool = False) -> Tuple[str, List[str]]:
    """
    Clean up the OCR content by removing special tags and tokens.
    
    Args:
        content: Raw OCR content
        extract_images: Whether to extract images from coordinate tags
        pdf_path: Path to original PDF (required if extract_images=True)
        remove_page_splits: Whether to remove page split markers
        
    Returns:
        Tuple of (cleaned_content, list_of_extracted_image_paths)
    """
    extracted_images = []
    
    if not content:
        return content, []
    
    # Remove end of sentence tokens
    if '<｜end▁of▁sentence｜>' in content:
        content = content.replace('<｜end▁of▁sentence｜>', '')
    
    # Extract images if requested
    if extract_images and pdf_path:
        content, num_images, extracted_images = extract_and_save_images(pdf_path, content)
        if num_images > 0:
            logger.info(f"Extracted {num_images} images from {pdf_path}")
    else:
        # Just remove image tags without extracting
        _, matches_images, _ = re_match_tags(content)
        for tag in matches_images:
            content = content.replace(tag, '', 1)
    
    # Get all non-image references
    _, _, matches_other = re_match_tags(content)
    
    # Remove other reference tags (text, table, title bounding boxes)
    for tag in matches_other:
        content = content.replace(tag, '')
    
    # Remove INCOMPLETE/TRUNCATED tags (where OCR output was cut off)
    # These look like: <|ref|>text<|/ref|><|det|>[[129, 803, 366,  (missing closing <|/det|>)
    # Match from <|ref|> or <|det|> to end of content if no closing tag
    content = re.sub(r'<\|ref\|>[^<]*$', '', content)  # Incomplete ref at end
    content = re.sub(r'<\|det\|>[^<]*$', '', content)  # Incomplete det at end
    # Also catch incomplete tags with partial data (det followed by coordinates but no close)
    content = re.sub(r'<\|ref\|>\w+<\|/ref\|><\|det\|>\[\[[\d\s,\.]*$', '', content)
    # Remove any orphaned opening tags
    content = re.sub(r'<\|ref\|>(?![^<]*<\|/ref\|>)', '', content)
    content = re.sub(r'<\|det\|>(?![^<]*<\|/det\|>)', '', content)
    
    # Remove page split markers if requested
    if remove_page_splits:
        content = re.sub(r'\n*<-+\s*Page\s*Split\s*-+>\n*', '\n\n', content, flags=re.IGNORECASE)
    
    # Replace special LaTeX-like symbols
    content = content.replace('\\coloneqq', ':=')
    content = content.replace('\\eqqcolon', '=:')
    
    # Clean up excessive newlines
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    content = content.replace('\n\n\n', '\n\n')
    
    return content.strip(), extracted_images


# =============================================================================
# File Management Functions
# =============================================================================


def get_file_hash(file_path):
    """Get MD5 hash of a file for caching"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def save_uploaded_file(pdf_file):
    """Save uploaded file to persistent storage and return the path"""
    if pdf_file is None:
        return None, None
    
    # Handle both string paths and file objects
    if isinstance(pdf_file, str):
        source_path = pdf_file
    elif hasattr(pdf_file, 'name'):
        source_path = pdf_file.name
    else:
        return None, None
    
    # Create a unique filename with timestamp
    original_name = Path(source_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{original_name}_{timestamp}.pdf"
    dest_path = UPLOAD_DIR / new_filename
    
    # Copy the file
    shutil.copy2(source_path, dest_path)
    logger.info(f"Saved uploaded file to: {dest_path}")
    
    return str(dest_path), new_filename


def save_result(filename, result_text, mode, prompt):
    """Save processing result to persistent storage"""
    if not result_text:
        return None
    
    # Create result filename
    base_name = Path(filename).stem if filename else "result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine suffix based on mode
    suffix_map = {"Markdown": "MD", "OCR": "OCR", "Custom Prompt": "CUSTOM"}
    suffix = suffix_map.get(mode, "OUT")
    
    result_filename = f"{base_name}_{suffix}.md"
    result_path = RESULTS_DIR / result_filename
    
    # Save the result
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    # Also save metadata
    meta_path = RESULTS_DIR / f"{base_name}_{suffix}_meta.json"
    metadata = {
        "original_file": filename,
        "mode": mode,
        "prompt": prompt,
        "timestamp": timestamp,
        "result_file": result_filename
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved result to: {result_path}")
    return str(result_path)


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


def check_for_cached_result(pdf_path, mode):
    """Check if we have a cached result for this file"""
    if not pdf_path:
        return None
    
    base_name = Path(pdf_path).stem
    suffix_map = {"Markdown": "MD", "OCR": "OCR", "Custom Prompt": "CUSTOM"}
    suffix = suffix_map.get(mode, "OUT")
    
    # Look for existing result
    result_path = RESULTS_DIR / f"{base_name}_{suffix}.md"
    if result_path.exists():
        with open(result_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def list_previous_uploads():
    """List previously uploaded files"""
    files = list(UPLOAD_DIR.glob("*.pdf"))
    return [str(f) for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]


def list_previous_results():
    """List previously saved results"""
    files = list(RESULTS_DIR.glob("*.md"))
    return [str(f) for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]


def add_to_queue(pdf_file, current_queue):
    """Add uploaded file(s) to the processing queue"""
    if pdf_file is None:
        return current_queue, format_queue_display(current_queue), "⚠️ No file selected"
    
    # Handle different input formats
    if isinstance(pdf_file, str):
        new_files = [pdf_file]
    elif isinstance(pdf_file, list):
        new_files = []
        for f in pdf_file:
            if isinstance(f, str):
                new_files.append(f)
            elif hasattr(f, 'name'):
                new_files.append(f.name)
    elif hasattr(pdf_file, 'name'):
        new_files = [pdf_file.name]
    else:
        return current_queue, format_queue_display(current_queue), "⚠️ Invalid file format"
    
    # Add each file to queue with page count
    added_count = 0
    for file_path in new_files:
        # Get page count
        page_count = client.get_pdf_page_count(file_path)
        filename = Path(file_path).name
        
        # Check if already in queue
        if not any(item['path'] == file_path for item in current_queue):
            current_queue.append({
                'path': file_path,
                'name': filename,
                'pages': page_count or '?',
                'status': 'pending'
            })
            added_count += 1
    
    status = f"✅ Added {added_count} file(s) to queue" if added_count > 0 else "ℹ️ File(s) already in queue"
    return current_queue, format_queue_display(current_queue), status


def format_queue_display(queue):
    """Format queue for display"""
    if not queue:
        return "📭 Queue is empty - upload files and click 'Add to Queue'"
    
    lines = ["📋 **Processing Queue:**", ""]
    for i, item in enumerate(queue, 1):
        status_icon = {
            'pending': '⏳',
            'processing': '🔄',
            'complete': '✅',
            'error': '❌'
        }.get(item['status'], '❓')
        pages = f"({item['pages']} pages)" if item['pages'] != '?' else ""
        lines.append(f"{i}. {status_icon} {item['name']} {pages}")
    
    return "\n".join(lines)


def clear_queue(current_queue):
    """Clear the processing queue"""
    return [], "📭 Queue is empty - drop files to add them", "🗑️ Queue cleared"


def process_queue(queue, processing_mode, custom_prompt_text, extract_images, remove_page_splits):
    """
    Process all files in the queue sequentially with real-time progress updates.
    This is a generator function that yields updates.
    
    Args:
        queue: List of queue items
        processing_mode: "Markdown", "OCR", or "Custom Prompt"
        custom_prompt_text: Custom prompt text (if mode is Custom Prompt)
        extract_images: Whether to extract images from detected regions
        remove_page_splits: Whether to remove page split markers
    
    Yields:
        Tuple of (queue, queue_display, progress_status, raw_result, rendered_result, result_files, images_zip)
    """
    global processing_stats, cancel_requested
    
    # Reset cancel flag at start
    cancel_requested = False
    
    if not queue:
        yield queue, format_queue_display(queue), "⚠️ Queue is empty", "", "", [], None
        return
    
    # Determine the prompt based on processing mode
    if processing_mode == "Markdown":
        prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    elif processing_mode == "OCR":
        prompt = '<image>\nFree OCR.'
    elif processing_mode == "Custom Prompt":
        if not custom_prompt_text.strip():
            yield queue, format_queue_display(queue), "⚠️ Please enter a custom prompt", "", "", None, None
            return
        prompt = custom_prompt_text
    else:
        yield queue, format_queue_display(queue), "⚠️ Invalid processing mode", "", "", None, None
        return

    # Check API health first
    is_healthy, health_msg = client.check_health()
    if not is_healthy:
        yield queue, format_queue_display(queue), f"⚠️ API Error: {health_msg}", "", "", None, None
        return

    total_files = len(queue)
    all_results = []
    all_results_raw = []
    all_result_paths = []  # Track all result files
    all_extracted_images = []  # Collect all extracted images for zip
    last_images_zip = None
    
    for file_idx, item in enumerate(queue):
        # Check for cancel request between files
        if cancel_requested:
            progress_status = f"⚠️ Cancelled after {file_idx}/{total_files} files"
            yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths, last_images_zip
            return
        
        file_num = file_idx + 1
        filename = item['name']
        file_path = item['path']
        pages = item['pages']
        
        # Update status to processing
        queue[file_idx]['status'] = 'processing'
        
        # Use adaptive timing estimate
        avg_time = processing_stats["avg_seconds_per_page"]
        
        # Initial status for this file
        estimated_time = pages * avg_time if pages != '?' else 60
        progress_status = f"🔄 [{file_num}/{total_files}] {filename} - Starting... (ETA: ~{int(estimated_time)}s)"
        yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths if all_result_paths else [], last_images_zip
        
        # Save the uploaded file persistently
        saved_path, saved_filename = save_uploaded_file(file_path)
        if not saved_path:
            queue[file_idx]['status'] = 'error'
            progress_status = f"❌ [{file_num}/{total_files}] {filename} - Failed to save"
            yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths if all_result_paths else [], last_images_zip
            continue
        
        # Get page count if we don't have it
        if pages == '?':
            pages = client.get_pdf_page_count(saved_path) or 50
            queue[file_idx]['pages'] = pages
        
        # Check for cached result first
        cached = check_for_cached_result(saved_path, processing_mode)
        if cached:
            queue[file_idx]['status'] = 'complete'
            cleaned_result, extracted_imgs = clean_content(cached, extract_images=extract_images, pdf_path=saved_path, remove_page_splits=remove_page_splits)
            all_extracted_images.extend(extracted_imgs)
            result_path = save_result(saved_filename, cleaned_result, processing_mode, prompt)
            all_result_paths.append(result_path)
            all_results_raw.append(f"## {filename}\n\n{cleaned_result}")
            all_results.append(cleaned_result)
            
            # Create images zip if we have images
            if extracted_imgs:
                last_images_zip = create_images_zip(extracted_imgs, saved_filename)
            
            progress_status = f"✅ [{file_num}/{total_files}] {filename} - 100% (cached)"
            yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths, last_images_zip
            continue
        
        # Calculate timeout using adaptive timing
        estimated_time = pages * avg_time
        timeout = max(300, int(estimated_time * 1.5))
        
        # Start processing with progress simulation
        start_time = time.time()
        
        # Create a thread to do the actual processing
        result_container = {'success': None, 'result': None, 'pages': 0}
        
        def do_processing():
            result_container['success'], result_container['result'], result_container['pages'] = \
                client.process_pdf(saved_path, prompt, timeout=timeout)
        
        process_thread = threading.Thread(target=do_processing)
        process_thread.start()
        
        # While processing, yield progress updates with adaptive timing
        while process_thread.is_alive():
            # Check for cancel (note: can't stop mid-file, only between files)
            if cancel_requested:
                # Just continue processing current file, will check cancel after
                pass
            
            elapsed = time.time() - start_time
            # Recalculate estimate based on current average
            current_avg = processing_stats["avg_seconds_per_page"]
            current_estimate = pages * current_avg if pages else current_avg
            
            # Determine progress and adjust avg when we hit/past 100%
            if current_estimate > 0:
                raw_pct = (elapsed / current_estimate) * 100
            else:
                raw_pct = 50
            at_or_over = raw_pct >= 100
            progress_pct = min(int(raw_pct), 100)
            
            if at_or_over and pages:
                # Recalibrate avg so pages * new_avg == elapsed (stop at 100%)
                adjusted_avg = elapsed / pages
                processing_stats["avg_seconds_per_page"] = adjusted_avg
                remaining = 0
                eta_str = "ETA: ~0s (recalibrated)"
            else:
                remaining = max(0, current_estimate - elapsed)
                eta_str = f"ETA: ~{int(remaining)}s"
            
            progress_status = f"🔄 [{file_num}/{total_files}] {filename} - {progress_pct}% | {int(elapsed)}s elapsed | {eta_str} | ~{processing_stats['avg_seconds_per_page']:.1f}s/page"
            yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths if all_result_paths else [], last_images_zip
            time.sleep(2)
        
        process_thread.join()
        
        elapsed_time = time.time() - start_time
        
        if result_container['success']:
            queue[file_idx]['status'] = 'complete'
            
            # Update adaptive timing stats
            actual_pages = result_container['pages'] or pages
            processing_stats["total_pages_processed"] += actual_pages
            processing_stats["total_time_spent"] += elapsed_time
            if processing_stats["total_pages_processed"] > 0:
                processing_stats["avg_seconds_per_page"] = (
                    processing_stats["total_time_spent"] / processing_stats["total_pages_processed"]
                )
            logger.info(f"Updated avg time/page: {processing_stats['avg_seconds_per_page']:.2f}s")
            
            # Apply post-processing: clean tags, optionally extract images
            raw_result = result_container['result']
            cleaned_result, extracted_imgs = clean_content(raw_result, extract_images=extract_images, pdf_path=saved_path, remove_page_splits=remove_page_splits)
            all_extracted_images.extend(extracted_imgs)
            
            result_path = save_result(saved_filename, cleaned_result, processing_mode, prompt)
            all_result_paths.append(result_path)
            all_results_raw.append(f"## {filename}\n\n{cleaned_result}")
            all_results.append(cleaned_result)
            
            # Create images zip if we have images for this file
            if extracted_imgs:
                last_images_zip = create_images_zip(extracted_imgs, saved_filename)
            
            progress_status = f"✅ [{file_num}/{total_files}] {filename} - 100% | {int(elapsed_time)}s | {elapsed_time/actual_pages:.1f}s/page"
        else:
            queue[file_idx]['status'] = 'error'
            progress_status = f"❌ [{file_num}/{total_files}] {filename} - Failed: {result_container['result'][:100]}"
        
        yield queue, format_queue_display(queue), progress_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths, last_images_zip
    
    # Final summary
    complete_count = sum(1 for item in queue if item['status'] == 'complete')
    error_count = sum(1 for item in queue if item['status'] == 'error')
    final_status = f"🏁 Done! {complete_count}/{total_files} completed"
    if error_count > 0:
        final_status += f", {error_count} failed"
    final_status += f" | Avg: {processing_stats['avg_seconds_per_page']:.1f}s/page"
    if all_extracted_images:
        final_status += f" | {len(all_extracted_images)} images extracted"
    
    yield queue, format_queue_display(queue), final_status, "\n\n".join(all_results_raw), "\n\n---\n\n".join(all_results), all_result_paths, last_images_zip


def load_previous_result(result_path):
    """Load a previously saved result"""
    if not result_path or not os.path.exists(result_path):
        return "No result selected", "", ""
    
    with open(result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return f"📁 Loaded: {result_path}", content, content


def check_api_status():
    """Check and return API status"""
    is_healthy, msg = client.check_health()
    return msg


def update_prompt_visibility(processing_mode):
    """Show/hide custom prompt textbox based on processing mode"""
    is_custom = (processing_mode == "Custom Prompt")
    return gr.update(visible=is_custom), gr.update(visible=is_custom)


def refresh_file_lists():
    """Refresh the lists of uploads and results"""
    results = list_previous_results()
    return gr.update(choices=results)


def request_cancel():
    """Request cancellation of current processing"""
    global cancel_requested
    cancel_requested = True
    return "⚠️ Cancel requested... will stop after current file"


def clear_completed_from_queue(current_queue):
    """Remove completed items from the queue"""
    new_queue = [item for item in current_queue if item['status'] not in ['complete', 'error']]
    cleared_count = len(current_queue) - len(new_queue)
    msg = f"🗑️ Removed {cleared_count} completed/failed items" if cleared_count > 0 else "ℹ️ No completed items to remove"
    return new_queue, format_queue_display(new_queue), msg


# Create the Gradio interface
custom_css = """
.scrollable-markdown {
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    background-color: #fafafa;
}
.scrollable-markdown::-webkit-scrollbar {
    width: 8px;
}
.scrollable-markdown::-webkit-scrollbar-thumb {
    background-color: #888;
    border-radius: 4px;
}
"""

with gr.Blocks(title="DeepSeek-OCR PDF Converter", theme=gr.themes.Soft(), css=custom_css) as demo:
    # State to hold the queue
    queue_state = gr.State([])
    
    gr.Markdown("""
    # 📄 DeepSeek-OCR PDF Converter (Enhanced)
    
    Convert PDF documents to Markdown or extract text using DeepSeek-OCR API.
    
    **Features:**
    - 📋 Queue system - add files one by one or in batches (even while processing!)
    - 🔄 Real-time progress with adaptive time estimates & ETA
    - ⏹️ Cancel button to stop between files
    - 🧹 Auto-cleans custom tags (`<|ref|>`, `<|det|>`, etc.)
    - 🖼️ Optional image extraction with ZIP download
    - 💾 Auto-saves uploads and results to disk
    - ⏱️ Extended timeout for large documents
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # API Status
            api_status = gr.Textbox(
                label="API Status",
                value=check_api_status(),
                interactive=False,
                lines=1
            )
            refresh_btn = gr.Button("🔄 Refresh API", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("### 📤 Drop Files (auto-adds to queue)")
            
            # File upload - files are auto-added to queue on drop
            # Using file_count="single" and handling each drop separately avoids clear issues
            pdf_input = gr.File(
                label="Drop PDFs here - they'll be added to the queue automatically",
                file_types=[".pdf"],
                file_count="multiple",
                type="file"  # Gradio 3.x uses 'file' not 'filepath'
            )
            
            # Queue management buttons
            with gr.Row():
                clear_queue_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                clear_completed_btn = gr.Button("✅ Clear Done", size="sm")
            
            # Queue display
            queue_display = gr.Markdown(
                value="📭 Queue is empty - drop PDF files above",
                label="Queue"
            )
            
            gr.Markdown("---")
            gr.Markdown("### ⚙️ Settings")
            
            # Processing mode
            processing_mode = gr.Radio(
                choices=["Markdown", "OCR", "Custom Prompt"],
                value="Markdown",
                label="Processing Mode"
            )
            
            # Post-processing options
            extract_images_checkbox = gr.Checkbox(
                label="🖼️ Extract Images",
                value=False,
                info="Extract images from detected regions (download as ZIP)"
            )
            remove_page_splits_checkbox = gr.Checkbox(
                label="📄 Remove Page Splits",
                value=True,
                info="Remove '<--- Page Split --->' markers from output"
            )
            
            # Custom prompt input (initially hidden)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt",
                placeholder="<image>\nYour custom instructions...",
                lines=2,
                visible=False
            )
            load_prompt_btn = gr.Button("📂 Load from YAML", size="sm", visible=False)
            
            # Process and Cancel buttons
            with gr.Row():
                process_btn = gr.Button("🚀 Process Queue", variant="primary", size="lg")
                cancel_btn = gr.Button("⏹️ Cancel", variant="stop", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### 📂 Previous Results")
            prev_results = gr.Dropdown(
                label="Load Previous Result",
                choices=list_previous_results(),
                interactive=True
            )
            with gr.Row():
                load_result_btn = gr.Button("📥 Load", size="sm")
                refresh_lists_btn = gr.Button("🔄 Refresh", size="sm")
        
        with gr.Column(scale=2):
            # Progress status - single line, prominent
            progress_status = gr.Textbox(
                label="Progress",
                value="Ready - drop files and click 'Process Queue'",
                interactive=False,
                lines=1
            )
            
            # Result output with tabs for raw and rendered markdown
            with gr.Tabs():
                with gr.Tab("📖 Rendered"):
                    # Scrollable markdown container
                    result_markdown = gr.Markdown(
                        value="*Results will appear here after processing...*",
                        label="Rendered Markdown",
                        elem_classes=["scrollable-markdown"]
                    )
                with gr.Tab("📝 Raw"):
                    result_raw = gr.Textbox(
                        label="Raw Markdown",
                        interactive=False,
                        lines=25,
                        max_lines=50  # Limited to make it scrollable
                    )
            
            # Downloads section
            gr.Markdown("### 📥 Downloads")
            with gr.Row():
                result_files = gr.File(
                    label="Markdown Results",
                    file_count="multiple",  # Allow multiple files
                    visible=True
                )
                images_zip_file = gr.File(
                    label="Extracted Images (ZIP)",
                    visible=True
                )
    
    gr.Markdown("""
    ---
    ### 📝 Quick Start
    
    1. **Drop Files**: Drop PDF files - they're automatically added to the queue
    2. **Configure**: Choose processing mode and enable image extraction if needed
    3. **Process**: Click "Process Queue" - watch progress with adaptive time estimates
    4. **Cancel**: Use "Cancel" to stop after current file (can't stop mid-file)
    5. **Download**: Results auto-saved; download markdown and/or images ZIP
    
    ### ⏱️ Processing Time
    - Initial estimate: ~15 seconds per page
    - Adapts based on actual processing speed
    - Progress shows real-time updates with ETA
    """)
    
    # Event handlers
    refresh_btn.click(
        fn=check_api_status,
        outputs=api_status
    )
    
    processing_mode.change(
        fn=update_prompt_visibility,
        inputs=processing_mode,
        outputs=[custom_prompt_input, load_prompt_btn]
    )
    
    load_prompt_btn.click(
        fn=load_custom_prompt,
        outputs=custom_prompt_input
    )
    
    # Auto-add files to queue when dropped - FIXED: don't clear the file input
    def handle_file_drop(files, current_queue):
        """Automatically add files to queue when dropped."""
        if files is None:
            return current_queue, format_queue_display(current_queue), gr.update()
        new_queue, display, message = add_to_queue(files, current_queue)
        # Return gr.update() for pdf_input to NOT clear it (keeps the drop zone visible)
        return new_queue, display, gr.update(value=None)
    
    pdf_input.change(
        fn=handle_file_drop,
        inputs=[pdf_input, queue_state],
        outputs=[queue_state, queue_display, pdf_input]
    )
    
    def handle_clear_queue(current_queue):
        """Clear the queue."""
        cleared, display, message = clear_queue(current_queue)
        return cleared, display
    
    clear_queue_btn.click(
        fn=handle_clear_queue,
        inputs=queue_state,
        outputs=[queue_state, queue_display]
    )
    
    # Clear completed items button
    clear_completed_btn.click(
        fn=clear_completed_from_queue,
        inputs=queue_state,
        outputs=[queue_state, queue_display, progress_status]
    )
    
    # Cancel button
    cancel_btn.click(
        fn=request_cancel,
        outputs=progress_status
    )
    
    # Process queue with generator for real-time updates
    # process_queue yields: queue, queue_display, progress_status, raw_result, rendered_result, result_paths, images_zip
    process_btn.click(
        fn=process_queue,
        inputs=[queue_state, processing_mode, custom_prompt_input, extract_images_checkbox, remove_page_splits_checkbox],
        outputs=[queue_state, queue_display, progress_status, result_raw, result_markdown, result_files, images_zip_file]
    )
    
    load_result_btn.click(
        fn=load_previous_result,
        inputs=prev_results,
        outputs=[progress_status, result_raw, result_markdown]
    )
    
    refresh_lists_btn.click(
        fn=refresh_file_lists,
        outputs=prev_results
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
    
    logger.info(f"📁 Uploads will be saved to: {UPLOAD_DIR.absolute()}")
    logger.info(f"📁 Results will be saved to: {RESULTS_DIR.absolute()}")
    
    # Enable queue for generator functions (real-time progress)
    demo.queue()
    
    # Launch Gradio
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True
    )
