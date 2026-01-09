#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface (Async Version)

A web interface for converting PDF files to Markdown or OCR text using the async DeepSeek OCR API.
Features:
- Async job submission - upload and get a job handle immediately
- Progress tracking - poll for status and progress updates
- Reconnection support - check status of previously submitted jobs
- Result download - fetch completed results
- Job history - view all submitted jobs
"""

import gradio as gr
import requests
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging

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
POLL_INTERVAL = 2  # seconds

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class AsyncOCRClient:
    """Client for interacting with async DeepSeek OCR API"""
    
    def __init__(self, api_base_url=API_BASE_URL):
        self.api_base_url = api_base_url
        self.health_endpoint = f"{api_base_url}/health"
        self.create_job_endpoint = f"{api_base_url}/jobs/create"
        self.job_status_endpoint = f"{api_base_url}/jobs/{{job_id}}"
        self.download_endpoint = f"{api_base_url}/jobs/{{job_id}}/download"
        self.metadata_endpoint = f"{api_base_url}/jobs/{{job_id}}/metadata"
        self.list_jobs_endpoint = f"{api_base_url}/jobs"
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return True, self._format_health_info(data)
            else:
                return False, f"❌ API returned status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Cannot connect to API: {str(e)}"
    
    def _format_health_info(self, data):
        """Format health check data"""
        info = f"✅ API is healthy\n"
        info += f"Model: {data.get('model_path', 'DeepSeek-OCR')}\n"
        info += f"CUDA Available: {data.get('cuda_available', False)}\n"
        info += f"Active Jobs: {data.get('active_jobs', 0)}\n"
        info += f"Queued Jobs: {data.get('queued_jobs', 0)}\n"
        info += f"Total Jobs: {data.get('total_jobs', 0)}"
        return info
    
    def create_job(self, pdf_path: str, prompt: str, timeout: int = 30):
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
            
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}\n{response.text}"
            
            result = response.json()
            if result.get('success'):
                job_id = result.get('job_id')
                message = result.get('message', 'Job created')
                logger.info(f"Job created: {job_id}")
                return job_id, message
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
    
    def download_result(self, job_id: str, output_path: str, timeout: int = 30):
        """Download job result"""
        try:
            url = self.download_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code != 200:
                return False, f"Error: Status code {response.status_code}\n{response.text}"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Result downloaded to {output_path}")
            return True, f"Result saved to {output_path}"
        
        except Exception as e:
            logger.error(f"Error downloading result: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_metadata(self, job_id: str, timeout: int = 10):
        """Get job metadata"""
        try:
            url = self.metadata_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}"
            
            return response.json(), None
        
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def list_jobs(self, limit: int = 100, timeout: int = 10):
        """List recent jobs"""
        try:
            response = requests.get(
                self.list_jobs_endpoint,
                params={'limit': limit},
                timeout=timeout
            )
            
            if response.status_code != 200:
                return [], f"Error: Status code {response.status_code}"
            
            result = response.json()
            return result.get('jobs', []), None
        
        except Exception as e:
            logger.error(f"Error listing jobs: {str(e)}")
            return [], f"Error: {str(e)}"


# Initialize client
client = AsyncOCRClient()


# =============================================================================
# Gradio Interface Functions
# =============================================================================

def check_api_health():
    """Check API health"""
    is_healthy, message = client.check_health()
    return message


def upload_and_create_job(pdf_file, prompt_choice, custom_prompt):
    """Upload PDF and create a job"""
    if pdf_file is None:
        return "❌ Please select a PDF file", "", ""
    
    # Determine prompt
    if prompt_choice == "Markdown (Default)":
        prompt = "<image>Convert the content of the image to markdown."
    elif prompt_choice == "OCR Only":
        prompt = "<image>Extract all text from the image."
    elif prompt_choice == "Custom":
        if not custom_prompt or not custom_prompt.strip():
            return "❌ Please provide a custom prompt", "", ""
        prompt = custom_prompt
    else:
        prompt = "<image>Convert the content of the image to markdown."
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(pdf_file.name).name
    save_path = UPLOAD_DIR / f"{timestamp}_{filename}"
    
    try:
        # Copy uploaded file
        import shutil
        shutil.copy(pdf_file.name, save_path)
        
        # Create job
        job_id, message = client.create_job(str(save_path), prompt)
        
        if job_id:
            status_msg = f"✅ Job Created!\n\nJob ID: {job_id}\n{message}\n\nProcessing started..."
            return status_msg, job_id, "Processing..."
        else:
            return f"❌ Job creation failed:\n{message}", "", ""
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


def check_job_status(job_id):
    """Check status of a job"""
    if not job_id or not job_id.strip():
        return "❌ Please provide a job ID", "", 0
    
    status_data, error = client.get_job_status(job_id)
    
    if error:
        return f"❌ {error}", "", 0
    
    # Format status message
    status = status_data.get('status', 'unknown')
    progress = status_data.get('progress', 0)
    processed_pages = status_data.get('processed_pages', 0)
    total_pages = status_data.get('total_pages', 0)
    filename = status_data.get('filename', 'Unknown')
    
    msg = f"📊 Job Status\n\n"
    msg += f"Job ID: {job_id}\n"
    msg += f"Filename: {filename}\n"
    msg += f"Status: {status.upper()}\n"
    msg += f"Progress: {progress:.1f}%\n"
    msg += f"Pages: {processed_pages}/{total_pages}\n"
    
    if status == 'completed':
        msg += "\n✅ Job completed! You can now download the result."
    elif status == 'failed':
        error = status_data.get('error', 'Unknown error')
        msg += f"\n❌ Job failed: {error}"
    elif status == 'processing':
        msg += "\n⏳ Processing in progress..."
    elif status == 'queued':
        msg += "\n⏳ Job is queued, waiting to start..."
    
    # Determine download button state
    download_enabled = (status == 'completed')
    
    return msg, gr.update(interactive=download_enabled), progress


def download_job_result(job_id):
    """Download completed job result"""
    if not job_id or not job_id.strip():
        return "❌ Please provide a job ID", "", ""
    
    # First check if job is completed
    status_data, error = client.get_job_status(job_id)
    if error:
        return f"❌ {error}", "", ""
    
    if status_data.get('status') != 'completed':
        return f"❌ Job not completed. Status: {status_data.get('status')}", "", ""
    
    # Get metadata
    metadata, _ = client.get_metadata(job_id)
    
    # Download result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = status_data.get('filename', 'result')
    output_filename = f"{Path(filename).stem}_{timestamp}_MD.md"
    output_path = RESULTS_DIR / output_filename
    
    success, message = client.download_result(job_id, str(output_path), timeout=60)
    
    if success:
        # Read the result to display
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Format metadata
        meta_str = ""
        if metadata:
            meta_str = f"\n\n📋 Metadata:\n"
            meta_str += f"Total Pages: {metadata.get('total_pages', 'N/A')}\n"
            meta_str += f"Processing Time: {metadata.get('processing_time_seconds', 0):.1f}s\n"
            meta_str += f"Processed At: {metadata.get('processed_at', 'N/A')}\n"
        
        msg = f"✅ Result downloaded successfully!\n{message}{meta_str}"
        return msg, content, str(output_path)
    else:
        return f"❌ Download failed:\n{message}", "", ""


def auto_poll_status(job_id, max_polls=1800):
    """Auto-poll job status until completion (max 1 hour with 2s intervals)"""
    if not job_id or not job_id.strip():
        return "❌ Please provide a job ID", "", 0
    
    for i in range(max_polls):
        status_data, error = client.get_job_status(job_id)
        
        if error:
            return f"❌ {error}", "", 0
        
        status = status_data.get('status')
        
        if status in ['completed', 'failed', 'cancelled']:
            # Final status
            return check_job_status(job_id)
        
        # Update and wait
        time.sleep(POLL_INTERVAL)
        
        # Yield intermediate update
        yield check_job_status(job_id)
    
    return "⚠️ Polling timeout reached", "", 0


def list_recent_jobs():
    """List recent jobs"""
    jobs, error = client.list_jobs(limit=50)
    
    if error:
        return f"❌ {error}"
    
    if not jobs:
        return "No jobs found"
    
    # Format job list
    msg = f"📋 Recent Jobs ({len(jobs)})\n\n"
    
    for job in jobs:
        status_emoji = {
            'completed': '✅',
            'processing': '⏳',
            'queued': '⏰',
            'failed': '❌',
            'cancelled': '🚫'
        }.get(job.get('status'), '❓')
        
        msg += f"{status_emoji} {job.get('job_id')}\n"
        msg += f"   File: {job.get('filename')}\n"
        msg += f"   Status: {job.get('status')} ({job.get('progress', 0):.1f}%)\n"
        msg += f"   Pages: {job.get('processed_pages')}/{job.get('total_pages')}\n"
        
        # Show created time
        created_at = datetime.fromtimestamp(job.get('created_at', 0))
        msg += f"   Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return msg


def process_and_wait(pdf_file, prompt_choice, custom_prompt):
    """Upload, create job, and auto-poll until completion"""
    # First create the job
    status_msg, job_id, _ = upload_and_create_job(pdf_file, prompt_choice, custom_prompt)
    
    if not job_id:
        yield status_msg, job_id, "", 0, gr.update(interactive=False)
        return
    
    # Initial status
    yield status_msg, job_id, "", 0, gr.update(interactive=False)
    
    # Poll until complete
    max_polls = 1800  # 1 hour max
    for i in range(max_polls):
        time.sleep(POLL_INTERVAL)
        
        status_data, error = client.get_job_status(job_id)
        
        if error:
            yield f"❌ {error}", job_id, "", 0, gr.update(interactive=False)
            return
        
        status = status_data.get('status')
        progress = status_data.get('progress', 0)
        processed_pages = status_data.get('processed_pages', 0)
        total_pages = status_data.get('total_pages', 0)
        
        # Format status message
        msg = f"📊 Job Status\n\n"
        msg += f"Job ID: {job_id}\n"
        msg += f"Status: {status.upper()}\n"
        msg += f"Progress: {progress:.1f}%\n"
        msg += f"Pages: {processed_pages}/{total_pages}\n"
        
        if status == 'completed':
            msg += "\n✅ Job completed! Downloading result..."
            yield msg, job_id, "", progress, gr.update(interactive=True)
            
            # Auto-download
            download_msg, content, path = download_job_result(job_id)
            final_msg = msg + "\n\n" + download_msg
            yield final_msg, job_id, content, progress, gr.update(interactive=True)
            return
        
        elif status == 'failed':
            error = status_data.get('error', 'Unknown error')
            msg += f"\n❌ Job failed: {error}"
            yield msg, job_id, "", progress, gr.update(interactive=False)
            return
        
        elif status == 'cancelled':
            msg += "\n🚫 Job was cancelled"
            yield msg, job_id, "", progress, gr.update(interactive=False)
            return
        
        else:
            msg += "\n⏳ Processing..."
            yield msg, job_id, "", progress, gr.update(interactive=False)
    
    yield "⚠️ Polling timeout reached", job_id, "", 0, gr.update(interactive=False)


# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="DeepSeek OCR - Async", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 DeepSeek OCR - Async Processing")
    gr.Markdown("Upload PDFs, get a job handle, and track progress. Perfect for long documents!")
    
    # API Health Check
    with gr.Accordion("🏥 API Health Check", open=False):
        health_btn = gr.Button("Check API Health")
        health_output = gr.Textbox(label="API Status", lines=6)
        health_btn.click(check_api_health, outputs=health_output)
    
    # Main Tabs
    with gr.Tabs():
        # Tab 1: Quick Process (Submit and Wait)
        with gr.TabItem("⚡ Quick Process"):
            gr.Markdown("### Upload PDF and wait for completion")
            
            with gr.Row():
                with gr.Column(scale=1):
                    quick_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                    
                    quick_prompt_choice = gr.Radio(
                        choices=["Markdown (Default)", "OCR Only", "Custom"],
                        value="Markdown (Default)",
                        label="Processing Mode"
                    )
                    
                    quick_custom_prompt = gr.Textbox(
                        label="Custom Prompt (if selected)",
                        placeholder="<image>Your custom prompt here...",
                        lines=3
                    )
                    
                    quick_process_btn = gr.Button("🚀 Process and Wait", variant="primary")
                
                with gr.Column(scale=2):
                    quick_status = gr.Textbox(label="Status", lines=10)
                    quick_progress = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        label="Progress (%)",
                        interactive=False
                    )
                    quick_job_id = gr.Textbox(label="Job ID", interactive=False)
                    quick_download_btn = gr.Button("📥 Download Result", interactive=False)
            
            with gr.Row():
                quick_result = gr.Textbox(label="Result (Markdown)", lines=20)
            
            # Event handlers
            quick_process_btn.click(
                process_and_wait,
                inputs=[quick_pdf, quick_prompt_choice, quick_custom_prompt],
                outputs=[quick_status, quick_job_id, quick_result, quick_progress, quick_download_btn]
            )
            
            quick_download_btn.click(
                download_job_result,
                inputs=[quick_job_id],
                outputs=[quick_status, quick_result, gr.Textbox(visible=False)]
            )
        
        # Tab 2: Manual Control (Submit, Check, Download separately)
        with gr.TabItem("🎮 Manual Control"):
            gr.Markdown("### Full control over job lifecycle")
            
            with gr.Row():
                with gr.Column():
                    manual_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                    
                    manual_prompt_choice = gr.Radio(
                        choices=["Markdown (Default)", "OCR Only", "Custom"],
                        value="Markdown (Default)",
                        label="Processing Mode"
                    )
                    
                    manual_custom_prompt = gr.Textbox(
                        label="Custom Prompt (if selected)",
                        placeholder="<image>Your custom prompt here...",
                        lines=3
                    )
                    
                    manual_submit_btn = gr.Button("📤 Submit Job", variant="primary")
                
                with gr.Column():
                    manual_job_id = gr.Textbox(label="Job ID", placeholder="Enter job ID or submit a new job")
                    
                    with gr.Row():
                        manual_check_btn = gr.Button("🔍 Check Status")
                        manual_download_btn = gr.Button("📥 Download Result", interactive=False)
                    
                    manual_progress = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        label="Progress (%)",
                        interactive=False
                    )
            
            with gr.Row():
                manual_status = gr.Textbox(label="Status", lines=8)
            
            with gr.Row():
                manual_result = gr.Textbox(label="Result (Markdown)", lines=20)
            
            # Event handlers
            manual_submit_btn.click(
                upload_and_create_job,
                inputs=[manual_pdf, manual_prompt_choice, manual_custom_prompt],
                outputs=[manual_status, manual_job_id, gr.Textbox(visible=False)]
            )
            
            manual_check_btn.click(
                check_job_status,
                inputs=[manual_job_id],
                outputs=[manual_status, manual_download_btn, manual_progress]
            )
            
            manual_download_btn.click(
                download_job_result,
                inputs=[manual_job_id],
                outputs=[manual_status, manual_result, gr.Textbox(visible=False)]
            )
        
        # Tab 3: Job History
        with gr.TabItem("📜 Job History"):
            gr.Markdown("### View all recent jobs")
            
            history_refresh_btn = gr.Button("🔄 Refresh Job List")
            history_output = gr.Textbox(label="Recent Jobs", lines=30)
            
            history_refresh_btn.click(
                list_recent_jobs,
                outputs=history_output
            )
            
            # Auto-load on tab open
            demo.load(list_recent_jobs, outputs=history_output)


if __name__ == "__main__":
    logger.info("Starting Async OCR GUI...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
