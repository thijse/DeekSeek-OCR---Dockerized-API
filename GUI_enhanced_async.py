#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface (Enhanced Async)

A web interface for converting PDF files to Markdown using the async DeepSeek OCR API.
Features:
- Async job submission with progress tracking
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Queue system - add files one by one, process sequentially  
- Real-time progress updates with async polling
- Reconnect to jobs - resume checking status of submitted jobs
- Image extraction with ZIP download
- Auto-cleans custom tags
- File and result persistence
- Job history view
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
import fitz  # PyMuPDF
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
API_BASE_URL = os.environ.get('OCR_API_URL', 'http://localhost:8000')
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
IMAGES_DIR = Path("data/images")
POLL_INTERVAL = 2  # seconds

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Global cancel flag
cancel_requested = False

# Global queue state
file_queue = []
processing_status = {"current": "", "progress": 0, "is_processing": False}


class AsyncOCRClient:
    """Client for interacting with async DeepSeek OCR API"""
    
    def __init__(self, api_base_url=API_BASE_URL):
        self.api_base_url = api_base_url
        self.health_endpoint = f"{api_base_url}/health"
        self.create_job_endpoint = f"{api_base_url}/jobs/create"
        self.job_status_endpoint = f"{api_base_url}/jobs/{{job_id}}"
        self.download_endpoint = f"{api_base_url}/jobs/{{job_id}}/download"
        self.metadata_endpoint = f"{api_base_url}/jobs/{{job_id}}/metadata"
        # In single-job mode we do not use list/cancel endpoints
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return True, self._format_health_info(data), data.get("available", True)
            else:
                return False, f"❌ API returned status code: {response.status_code}", False
        except requests.exceptions.RequestException as e:
            return False, f"❌ Cannot connect to API: {str(e)}", False
    
    def _format_health_info(self, data):
        """Format health check data"""
        available = data.get("available", True)
        status_icon = "✅" if available else "⏳"
        info = f"{status_icon} API healthy | Available: {available}\n"
        info += f"Model: deepseek-ai/DeepSeek-OCR"
        current_job = data.get("current_job")
        if current_job:
            info += f"\nActive job: {current_job.get('job_id', '')} ({current_job.get('progress', 0):.0f}% )"
        return info
    
    def create_job(self, pdf_path: str, prompt: str, timeout: int = 120):
        """Create a new OCR job"""
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                data = {'prompt': prompt}
                
                logger.info(f"Creating job for {pdf_path}...")
                response = requests.post(
                    self.create_job_endpoint,
                    files=files,
                    data=data,
                    timeout=timeout
                )
            
            if response.status_code == 503:
                return None, "Server busy - already processing a job"
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}\n{response.text}"
            
            result = response.json()
            if result.get('success'):
                job_id = result.get('job_id')
                logger.info(f"Job created: {job_id}")
                return job_id, None
            else:
                return None, "Job creation failed"
        
        except Exception as e:
            logger.error(f"Error creating job: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def get_job_status(self, job_id: str, timeout: int = 10):
        """Get status of a job"""
        try:
            url = self.job_status_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 404:
                return None, "Job not found"
            elif response.status_code != 200:
                return None, f"Error: Status code {response.status_code}"
            
            return response.json(), None
        
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def download_result(self, job_id: str, timeout: int = 30):
        """Download job result as text"""
        try:
            url = self.download_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}\n{response.text}"
            
            logger.info(f"Result downloaded for job {job_id}")
            return response.content.decode('utf-8'), None
        
        except Exception as e:
            logger.error(f"Error downloading result: {str(e)}")
            return None, f"Error: {str(e)}"
    
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

    def check_available(self):
        """Return (available: bool, message: str)"""
        ok, msg, available = self.check_health()
        return available if ok else False, msg


# Initialize client
client = AsyncOCRClient()


# =============================================================================
# Post-Processing Functions  
# =============================================================================

def re_match_tags(text: str) -> Tuple[List, List, List]:
    """Match reference patterns in the text"""
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
    """Convert PDF pages to PIL Images"""
    images = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
    
    return images


def extract_and_save_images(pdf_path: str, content: str) -> Tuple[str, List[str]]:
    """Extract images from content based on coordinate tags"""
    extracted_paths = []
    
    pdf_images = pdf_to_images(pdf_path)
    if not pdf_images:
        _, matches_images, _ = re_match_tags(content)
        for tag in matches_images:
            content = content.replace(tag, '[Image]', 1)
        return content, []
    
    _, matches_images, _ = re_match_tags(content)
    total_extracted = 0
    
    for img_idx, img_tag in enumerate(matches_images):
        try:
            pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
            det_match = re.search(pattern, img_tag)
            
            if det_match:
                det_content = det_match.group(1)
                try:
                    coordinates = eval(det_content)
                    page_to_use = img_idx % len(pdf_images) if len(pdf_images) > 1 else 0
                    page_image = pdf_images[page_to_use]
                    image_width, image_height = page_image.size
                    
                    for points in coordinates:
                        x1, y1, x2, y2 = points
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        if x1 >= x2 or y1 >= y2:
                            continue
                        
                        cropped = page_image.crop((x1, y1, x2, y2))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_filename = f"{Path(pdf_path).stem}_img{total_extracted}_{timestamp}.jpg"
                        image_path = IMAGES_DIR / image_filename
                        cropped.save(image_path)
                        extracted_paths.append(str(image_path))
                        
                        encoded_filename = urllib.parse.quote(image_filename)
                        markdown_link = f"\n![Extracted Image](images/{encoded_filename})\n"
                        content = content.replace(img_tag, markdown_link, 1)
                        
                        total_extracted += 1
                        break
                except Exception as e:
                    logger.error(f"Error processing image coordinates: {str(e)}")
                    content = content.replace(img_tag, '[Image - extraction failed]', 1)
        except Exception as e:
            logger.error(f"Error extracting image: {str(e)}")
            content = content.replace(img_tag, '[Image - error]', 1)
    
    return content, extracted_paths


def create_images_zip(image_paths: List[str], pdf_filename: str) -> Optional[str]:
    """Create a zip file containing extracted images"""
    if not image_paths:
        return None
    
    base_name = Path(pdf_filename).stem
    zip_filename = f"{base_name}_images.zip"
    zip_path = RESULTS_DIR / zip_filename
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_path in image_paths:
                zipf.write(img_path, Path(img_path).name)
        
        logger.info(f"Created images zip: {zip_path} with {len(image_paths)} images")
        return str(zip_path)
    except Exception as e:
        logger.error(f"Error creating images zip: {str(e)}")
        return None


def clean_content(content: str, extract_images: bool = False, pdf_path: str = None, remove_page_splits: bool = False) -> Tuple[str, List[str]]:
    """Clean up the OCR content by removing special tags"""
    extracted_images = []
    
    if not content:
        return content, []
    
    if '<｜end▁of▁sentence｜>' in content:
        content = content.replace('<｜end▁of▁sentence｜>', '')
    
    if extract_images and pdf_path:
        content, extracted_images = extract_and_save_images(pdf_path, content)
    else:
        _, matches_images, _ = re_match_tags(content)
        for tag in matches_images:
            content = content.replace(tag, '', 1)
    
    _, _, matches_other = re_match_tags(content)
    for tag in matches_other:
        content = content.replace(tag, '')
    
    content = re.sub(r'<\|ref\|>[^<]*$', '', content)
    content = re.sub(r'<\|det\|>[^<]*$', '', content)
    content = re.sub(r'<\|ref\|>\w+<\|/ref\|><\|det\|>\[\[[\d\s,\.]*$', '', content)
    content = re.sub(r'<\|ref\|>(?![^<]*<\|/ref\|>)', '', content)
    content = re.sub(r'<\|det\|>(?![^<]*<\|/det\|>)', '', content)
    
    if remove_page_splits:
        content = re.sub(r'\n*<-+\s*Page\s*Split\s*-+>\n*', '\n\n', content, flags=re.IGNORECASE)
    
    content = content.replace('\\coloneqq', ':=')
    content = content.replace('\\eqqcolon', '=:')
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
    """Save uploaded PDF to uploads directory"""
    if pdf_file is None:
        return None
    
    try:
        source_path = Path(pdf_file.name if hasattr(pdf_file, 'name') else pdf_file)
        filename = source_path.name
        file_hash = get_file_hash(source_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{file_hash}_{filename}"
        dest_path = UPLOAD_DIR / new_filename
        shutil.copy(source_path, dest_path)
        logger.info(f"Saved uploaded file: {dest_path}")
        return str(dest_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None


def save_result(filename, result_text, mode, prompt, job_id=None, metadata=None):
    """Save processing result to results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem
    result_filename = f"{base_name}_{timestamp}_MD.md"
    result_path = RESULTS_DIR / result_filename
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    meta_filename = f"{base_name}_{timestamp}_MD_meta.json"
    meta_path = RESULTS_DIR / meta_filename
    
    meta_data = {
        "original_filename": filename,
        "processing_mode": mode,
        "prompt_used": prompt,
        "timestamp": timestamp,
        "result_file": result_filename,
        "job_id": job_id
    }
    if metadata:
        meta_data.update(metadata)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2)
    
    logger.info(f"Saved result: {result_path}")
    return str(result_path)


def load_custom_prompt():
    """Load custom prompt from YAML file"""
    yaml_path = Path("custom_prompt.yaml")
    if not yaml_path.exists():
        return "❌ custom_prompt.yaml not found"
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if 'prompt' in data:
                return data['prompt']
            else:
                return "❌ No 'prompt' key found in YAML"
    except Exception as e:
        return f"❌ Error loading YAML: {str(e)}"


def list_previous_results():
    """List all previous result files"""
    try:
        results = list(RESULTS_DIR.glob("*_MD.md"))
        results.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(r) for r in results[:50]]
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return []


# =============================================================================
# Queue Management Functions
# =============================================================================

def add_to_queue(pdf_files, current_queue):
    """Add files to the queue"""
    global file_queue
    
    if pdf_files is None:
        return format_queue_display(file_queue), gr.update()
    
    # Handle both single and multiple files
    files_to_add = pdf_files if isinstance(pdf_files, list) else [pdf_files]
    
    added_count = 0
    for pdf_file in files_to_add:
        if pdf_file is None:
            continue
        
        # Save uploaded file
        saved_path = save_uploaded_file(pdf_file)
        if saved_path:
            # Get page count
            page_count = client.get_pdf_page_count(saved_path)
            
            # Add to queue
            queue_item = {
                "path": saved_path,
                "filename": Path(saved_path).name,
                "pages": page_count or "?",
                "status": "⏳ Pending",
                "job_id": None
            }
            file_queue.append(queue_item)
            added_count += 1
    
    if added_count > 0:
        logger.info(f"Added {added_count} file(s) to queue")
    
    return format_queue_display(file_queue), gr.update(value=None)


def format_queue_display(queue):
    """Format queue for display"""
    if not queue:
        return "📭 Queue is empty - drop PDF files above"
    
    display = f"📋 **Queue ({len(queue)} file(s))**\n\n"
    for i, item in enumerate(queue, 1):
        status = item.get("status", "⏳ Pending")
        filename = item.get("filename", "Unknown")
        pages = item.get("pages", "?")
        job_id = item.get("job_id", "")
        
        display += f"{i}. **{filename}** ({pages} pages) - {status}\n"
        if job_id:
            display += f"   Job ID: `{job_id}`\n"
    
    return display


def clear_queue():
    """Clear the entire queue"""
    global file_queue
    file_queue = []
    return format_queue_display(file_queue)


def clear_completed():
    """Clear completed items from queue"""
    global file_queue
    file_queue = [item for item in file_queue if item.get("status") not in ["✅ Done", "❌ Failed"]]
    return format_queue_display(file_queue)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_queue(queue, processing_mode, custom_prompt_text, extract_images, remove_page_splits):
    """Process all files in the queue with async job management - ONE AT A TIME"""
    global cancel_requested, file_queue, processing_status
    
    cancel_requested = False
    
    if not file_queue:
        yield "📭 Queue is empty", "", format_queue_display(file_queue), None, None
        return
    
    # Determine prompt
    if processing_mode == "Markdown":
        prompt = "<image>Convert the content of the image to markdown."
    elif processing_mode == "OCR":
        prompt = "<image>Extract all text from the image."
    elif processing_mode == "Custom Prompt":
        prompt = custom_prompt_text
        if not prompt or not prompt.strip():
            yield "❌ Please provide a custom prompt", "", format_queue_display(file_queue), None, None
            return
    else:
        prompt = "<image>Convert the content of the image to markdown."
    
    total_files = len(file_queue)
    all_result_files = []
    all_image_zips = []

    def wait_for_server_available(max_wait: int = 120):
        """Wait for the server to become available (single-job mode)"""
        checks = max(1, int(max_wait / POLL_INTERVAL))
        for _ in range(checks):
            available, health_msg = client.check_available()
            if available:
                return True, health_msg
            time.sleep(POLL_INTERVAL)
        return False, "Server still busy after waiting"
    
    # Process files ONE AT A TIME sequentially
    for file_idx, queue_item in enumerate(file_queue):
        if cancel_requested:
            queue_item["status"] = "🚫 Cancelled"
            yield f"⏹️ Cancelled by user", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            break
        
        pdf_path = queue_item["path"]
        filename = queue_item["filename"]
        
        # Update status - PROCESSING THIS FILE NOW
        queue_item["status"] = "🚀 Submitting..."
        processing_status["current"] = filename
        processing_status["is_processing"] = True
        
        status_msg = f"[{file_idx+1}/{total_files}] 📤 Submitting: {filename}"
        yield status_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None

        # Wait for server availability (single-job server)
        available, wait_msg = wait_for_server_available()
        if not available:
            queue_item["status"] = "⏳ Busy"
            yield f"❌ {wait_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            break
        else:
            queue_item["status"] = "🚀 Submitting..."
            yield f"[{file_idx+1}/{total_files}] {wait_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
        
        # Create async job - ONLY FOR THIS FILE
        job_id, error = client.create_job(pdf_path, prompt)
        
        if error:
            queue_item["status"] = f"❌ Failed (submit)"
            yield f"❌ Job creation failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            continue
        
        queue_item["job_id"] = job_id
        queue_item["status"] = "⏳ Processing 0%..."
        yield f"[{file_idx+1}/{total_files}] ⏳ Job submitted: {job_id[:20]}...", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
        
        # WAIT FOR THIS JOB TO COMPLETE before moving to next file
        max_polls = 3600  # 2 hours max
        last_progress = -1
        
        for poll_count in range(max_polls):
            if cancel_requested:
                queue_item["status"] = "🚫 Cancelled"
                break
            
            time.sleep(POLL_INTERVAL)
            
            status_data, error = client.get_job_status(job_id)
            if error:
                queue_item["status"] = "❌ Error (status check)"
                yield f"❌ Status check failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
            
            status = status_data.get('status')
            progress = status_data.get('progress', 0)
            processed_pages = status_data.get('processed_pages', 0)
            total_pages = status_data.get('total_pages', 0)
            
            # Update progress display
            if progress != last_progress:
                queue_item["status"] = f"⏳ {progress:.0f}%"
                status_msg = f"[{file_idx+1}/{total_files}] {filename}: {progress:.0f}% ({processed_pages}/{total_pages} pages)"
                yield status_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                last_progress = progress
            
            if status == 'completed':
                # Download result from server
                result_text, error = client.download_result(job_id)
                
                if error:
                    queue_item["status"] = "❌ Download failed"
                    yield f"❌ Download failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                    break
                
                # Clean and process result
                cleaned_text, extracted_image_paths = clean_content(
                    result_text,
                    extract_images=extract_images,
                    pdf_path=pdf_path,
                    remove_page_splits=remove_page_splits
                )
                
                # Create images zip if needed
                zip_path = None
                if extract_images and extracted_image_paths:
                    zip_path = create_images_zip(extracted_image_paths, filename)
                    if zip_path:
                        all_image_zips.append(zip_path)
                
                # Save result to local file
                result_file_path = save_result(
                    filename,
                    cleaned_text,
                    processing_mode,
                    prompt,
                    job_id=job_id
                )
                all_result_files.append(result_file_path)
                
                queue_item["status"] = "✅ Done"
                
                result_msg = f"✅ [{file_idx+1}/{total_files}] Completed: {filename}"
                if extract_images and extracted_image_paths:
                    result_msg += f" + {len(extracted_image_paths)} images"
                
                # Return result with download files
                yield result_msg, cleaned_text, format_queue_display(file_queue), all_result_files, all_image_zips or None
                break
                
            elif status == 'failed':
                error_msg = status_data.get('error', 'Unknown error')
                queue_item["status"] = "❌ Failed"
                yield f"❌ Job failed: {error_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
                
            elif status == 'cancelled':
                queue_item["status"] = "🚫 Cancelled"
                yield f"🚫 Job was cancelled", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
        
        if poll_count >= max_polls - 1:
            queue_item["status"] = "⏱️ Timeout"
            yield f"⏱️ Timeout waiting for job completion", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
    
    processing_status["is_processing"] = False
    
    # Final summary
    completed = len([item for item in file_queue if item.get("status") == "✅ Done"])
    failed = len([item for item in file_queue if "❌" in item.get("status", "")])
    
    final_msg = f"🎉 Processing complete! {completed} successful, {failed} failed"
    yield final_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None


def load_previous_result(result_path):
    """Load a previous result file"""
    if not result_path:
        return "", "No file selected"
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, f"Loaded: {Path(result_path).name}"
    except Exception as e:
        return "", f"Error loading file: {str(e)}"


def check_api_status():
    """Check API health"""
    _, message, _ = client.check_health()
    return message


def update_prompt_visibility(processing_mode):
    """Show/hide custom prompt based on mode"""
    is_custom = (processing_mode == "Custom Prompt")
    return gr.update(visible=is_custom), gr.update(visible=is_custom)


def refresh_file_lists():
    """Refresh the list of previous results"""
    return gr.update(choices=list_previous_results())


def request_cancel():
    """Request cancellation of current processing"""
    global cancel_requested
    cancel_requested = True
    return "🚫 Cancellation requested - will stop after current file"


# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="DeepSeek OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 DeepSeek OCR")
    # gr.Markdown("""
    # **Features:**
    # - 📋 Queue system with async job management
    # - 🔄 Real-time progress tracking
    # - 🔌 Reconnection support - check old jobs anytime
    # - 🧹 Auto-cleans custom tags
    # - 🖼️ Optional image extraction with ZIP download
    # - 💾 Auto-saves uploads and results to disk
    # - ⏹️ Cancel anytime (client-side queue)
    # """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # API Status
            api_status = gr.Textbox(
                label="API Status",
                value=check_api_status(),
                interactive=False,
                lines=3
            )
            refresh_api_btn = gr.Button("🔄 Refresh API", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("### 📤 Add Files to Queue")
            
            # File upload
            pdf_input = gr.File(
                label="Drop PDFs here",
                file_types=[".pdf"],
                file_count="multiple",
                type="file"
            )
            
            # Queue management
            with gr.Row():
                clear_queue_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                clear_completed_btn = gr.Button("✅ Clear Done", size="sm")
            
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
                info="Extract images from detected regions"
            )
            remove_page_splits_checkbox = gr.Checkbox(
                label="📄 Remove Page Splits",
                value=True,
                info="Remove '<--- Page Split --->' markers"
            )
            
            # Custom prompt
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt",
                placeholder="<image>\nYour custom instructions...",
                lines=2,
                visible=False
            )
            load_prompt_btn = gr.Button("📂 Load from YAML", size="sm", visible=False)
            
            # Process and Cancel
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
            # Progress status
            progress_status = gr.Textbox(
                label="Progress",
                value="Ready - add files and click 'Process Queue'",
                interactive=False,
                lines=1
            )
            
            # Result output with tabs
            with gr.Tabs():
                with gr.TabItem("📄 Rendered"):
                    result_rendered = gr.Markdown(
                        label="Result (Rendered)",
                        value="*Results will appear here after processing...*",
                        elem_id="rendered-output"
                    )
                
                with gr.TabItem("📝 Raw Markdown"):
                    result_text = gr.Textbox(
                        label="Result (Raw Markdown)",
                        lines=30,
                        max_lines=100,
                        value="",
                        show_copy_button=True
                    )
            
            # Downloads section
            gr.Markdown("### 📥 Downloads")
            with gr.Row():
                result_files = gr.File(
                    label="Markdown Results",
                    file_count="multiple",
                    visible=True
                )
                images_zip_file = gr.File(
                    label="Extracted Images (ZIP)",
                    file_count="multiple",
                    visible=True
                )
    
    # gr.Markdown("""
    # ---
    # ### 📝 Quick Start
    
    # 1. **Drop Files**: Drop PDF files into the upload area
    # 2. **Configure**: Choose processing mode and enable options
    # 3. **Process**: Click "Process Queue" - files are processed **ONE AT A TIME sequentially**
    # 4. **Wait**: Each file completes before the next starts (no parallel processing)
    # 5. **Download**: Download markdown files and image ZIPs from the Downloads section
    
    # ### ⏱️ Processing
    # - Files processed sequentially (one at a time)
    # - Real-time progress tracking for each file
    # - Results auto-saved locally and downloadable
    # """)
    
    # Event handlers
    refresh_api_btn.click(
        check_api_status,
        outputs=api_status
    )
    
    pdf_input.change(
        add_to_queue,
        inputs=[pdf_input, queue_display],
        outputs=[queue_display, pdf_input]
    )
    
    clear_queue_btn.click(
        clear_queue,
        outputs=queue_display
    )
    
    clear_completed_btn.click(
        clear_completed,
        outputs=queue_display
    )
    
    processing_mode.change(
        update_prompt_visibility,
        inputs=processing_mode,
        outputs=[custom_prompt_input, load_prompt_btn]
    )
    
    load_prompt_btn.click(
        load_custom_prompt,
        outputs=custom_prompt_input
    )
    
    process_btn.click(
        process_queue,
        inputs=[
            queue_display,
            processing_mode,
            custom_prompt_input,
            extract_images_checkbox,
            remove_page_splits_checkbox
        ],
        outputs=[progress_status, result_text, queue_display, result_files, images_zip_file]
    )
    
    result_text.change(
        lambda x: x,
        inputs=result_text,
        outputs=result_rendered
    )
    
    cancel_btn.click(
        request_cancel,
        outputs=progress_status
    )
    
    load_result_btn.click(
        load_previous_result,
        inputs=prev_results,
        outputs=[result_text, progress_status]
    )
    
    refresh_lists_btn.click(
        refresh_file_lists,
        outputs=prev_results
    )



if __name__ == "__main__":
    logger.info("Starting Enhanced Async OCR GUI...")
    demo.queue()  # Enable queue for generator functions
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
