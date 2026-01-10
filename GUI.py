#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface

A web interface for converting PDF files to Markdown using the DeepSeek OCR API.
Features:
- Async job submission with progress tracking
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Queue system - add files one by one, process sequentially  
- Real-time progress updates with async polling
- Image extraction with ZIP download
- Auto-cleans custom tags
- File and result persistence
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
