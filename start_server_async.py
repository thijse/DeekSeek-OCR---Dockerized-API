#!/usr/bin/env python3
"""
DeepSeek-OCR vLLM Server with Async Job Management
FastAPI wrapper for DeepSeek-OCR with vLLM backend and job queue
"""

import os
import sys
import asyncio
import io
import tempfile
import uuid
import time
import json
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

# Add current directory to Python path
sys.path.insert(0, '/app/DeepSeek-OCR-vllm')

# Set environment variables for vLLM compatibility
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Import DeepSeek-OCR components
from config import INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE, MAX_CONCURRENCY, NUM_WORKERS
MODEL_PATH = os.environ.get('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR')
from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Register the custom model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Job storage configuration
JOB_STORAGE_DIR = Path("/app/jobs")
JOB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_STORAGE_MB = int(os.environ.get('MAX_JOB_STORAGE_MB', '10240'))  # 10GB default
MAX_JOB_COUNT = int(os.environ.get('MAX_JOB_COUNT', '1000'))
MAX_JOB_AGE_HOURS = int(os.environ.get('MAX_JOB_AGE_HOURS', '168'))  # 7 days default

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    job_id: str
    filename: str
    status: JobStatus
    progress: float  # 0.0 to 100.0
    total_pages: int
    processed_pages: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_path: Optional[str] = None
    metadata_path: Optional[str] = None

class JobManager:
    """Manages OCR job queue and persistent storage"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.job_order = OrderedDict()  # For LRU cleanup
        self.current_job_id: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # Restore jobs from disk on startup
        self._restore_jobs()
    
    def _restore_jobs(self):
        """Restore job state from disk on startup"""
        for job_dir in JOB_STORAGE_DIR.iterdir():
            if job_dir.is_dir():
                metadata_file = job_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Convert dict back to Job object
                        job = Job(**data)
                        
                        # Reset processing/queued jobs to failed on restart
                        if job.status in [JobStatus.PROCESSING, JobStatus.QUEUED]:
                            job.status = JobStatus.FAILED
                            job.error = "Server restarted during processing"
                        
                        self.jobs[job.job_id] = job
                        self.job_order[job.job_id] = time.time()
                        
                    except Exception as e:
                        print(f"Error restoring job from {job_dir}: {e}")
    
    def _save_job_metadata(self, job: Job):
        """Save job metadata to disk"""
        job_dir = JOB_STORAGE_DIR / job.job_id
        job_dir.mkdir(exist_ok=True)
        
        metadata_file = job_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(job), f, indent=2)
    
    def _cleanup_old_jobs(self):
        """Remove old jobs to maintain storage limits"""
        # Get all jobs sorted by age (oldest first)
        sorted_jobs = sorted(
            self.jobs.items(),
            key=lambda x: x[1].created_at
        )
        
        # Check storage size
        total_size_mb = sum(
            sum(f.stat().st_size for f in Path(JOB_STORAGE_DIR / job_id).rglob('*') if f.is_file())
            for job_id in self.jobs.keys()
        ) / (1024 * 1024)
        
        # Check job age
        current_time = time.time()
        max_age_seconds = MAX_JOB_AGE_HOURS * 3600
        
        jobs_to_remove = []
        
        # Remove by age
        for job_id, job in sorted_jobs:
            if current_time - job.created_at > max_age_seconds:
                jobs_to_remove.append(job_id)
        
        # Remove by count
        if len(self.jobs) > MAX_JOB_COUNT:
            for job_id, job in sorted_jobs[:len(self.jobs) - MAX_JOB_COUNT]:
                if job_id not in jobs_to_remove:
                    jobs_to_remove.append(job_id)
        
        # Remove by storage size
        while total_size_mb > MAX_STORAGE_MB and len(sorted_jobs) > len(jobs_to_remove):
            oldest_job = None
            for job_id, job in sorted_jobs:
                if job_id not in jobs_to_remove:
                    oldest_job = job_id
                    break
            
            if oldest_job:
                jobs_to_remove.append(oldest_job)
                # Recalculate size
                job_dir = JOB_STORAGE_DIR / oldest_job
                if job_dir.exists():
                    job_size = sum(f.stat().st_size for f in job_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                    total_size_mb -= job_size
            else:
                break
        
        # Actually remove the jobs
        for job_id in jobs_to_remove:
            self._remove_job(job_id)
        
        if jobs_to_remove:
            print(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def _remove_job(self, job_id: str):
        """Remove a job and its files"""
        if job_id in self.jobs:
            del self.jobs[job_id]
        if job_id in self.job_order:
            del self.job_order[job_id]
        
        # Remove directory
        job_dir = JOB_STORAGE_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
    
    async def create_job(self, filename: str, pdf_data: bytes, total_pages: int) -> str:
        """Create a new job and save the input file"""
        async with self.lock:
            # Cleanup before creating new job
            self._cleanup_old_jobs()
            
            # Generate unique job ID
            clean_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.')).rstrip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            job_id = f"{clean_filename}_{timestamp}_{unique_id}"
            
            # Create job directory
            job_dir = JOB_STORAGE_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Save input PDF
            input_path = job_dir / "input.pdf"
            with open(input_path, 'wb') as f:
                f.write(pdf_data)
            
            # Create job object
            job = Job(
                job_id=job_id,
                filename=filename,
                status=JobStatus.QUEUED,
                progress=0.0,
                total_pages=total_pages,
                processed_pages=0,
                created_at=time.time()
            )
            
            self.jobs[job_id] = job
            self.job_order[job_id] = time.time()
            self._save_job_metadata(job)
            
            return job_id
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    async def update_job_progress(self, job_id: str, processed_pages: int):
        """Update job progress"""
        async with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.processed_pages = processed_pages
                if job.total_pages > 0:
                    job.progress = (processed_pages / job.total_pages) * 100.0
                self._save_job_metadata(job)
    
    async def start_job(self, job_id: str):
        """Mark job as started"""
        async with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()
                self.current_job_id = job_id
                self._save_job_metadata(job)
    
    async def complete_job(self, job_id: str, result_data: str, metadata: Dict[str, Any]):
        """Mark job as completed and save results"""
        async with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = 100.0
                
                # Save result file
                job_dir = JOB_STORAGE_DIR / job_id
                result_path = job_dir / "result.md"
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(result_data)
                job.result_path = str(result_path)
                
                # Save metadata
                metadata_result_path = job_dir / "result_metadata.json"
                with open(metadata_result_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                job.metadata_path = str(metadata_result_path)
                
                self._save_job_metadata(job)
                self.current_job_id = None
    
    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        async with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = error
                self._save_job_metadata(job)
                self.current_job_id = None
    
    async def cancel_job(self, job_id: str):
        """Cancel a job"""
        async with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = time.time()
                    self._save_job_metadata(job)
                    if self.current_job_id == job_id:
                        self.current_job_id = None
    
    async def list_jobs(self, limit: int = 100) -> List[Job]:
        """List recent jobs"""
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return sorted_jobs[:limit]
    
    def get_job_file_path(self, job_id: str, file_type: str) -> Optional[Path]:
        """Get path to job file (input.pdf or result.md)"""
        job_dir = JOB_STORAGE_DIR / job_id
        if not job_dir.exists():
            return None
        
        if file_type == "input":
            return job_dir / "input.pdf"
        elif file_type == "result":
            return job_dir / "result.md"
        elif file_type == "metadata":
            return job_dir / "result_metadata.json"
        return None

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR Async API",
    description="High-performance OCR service with async job management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm = None
sampling_params = None
job_manager = JobManager()
processing_lock = asyncio.Lock()

# Response models
class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    progress: float
    total_pages: int
    processed_pages: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

class JobCreateResponse(BaseModel):
    success: bool
    job_id: str
    message: str

class JobListResponse(BaseModel):
    jobs: List[JobResponse]

# Model initialization (same as original)
def initialize_model():
    """Initialize the vLLM model"""
    global llm, sampling_params
    
    if llm is None:
        print("Initializing DeepSeek-OCR model...")
        
        # Initialize vLLM engine
        env_dtype = os.environ.get('DTYPE', '').strip().lower()
        cc = None
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                cc = props.major + props.minor / 10.0
        except Exception:
            cc = None

        if env_dtype in ('bf16', 'bfloat16'):
            selected_dtype = 'bfloat16'
        elif env_dtype in ('fp16', 'half', 'float16'):
            selected_dtype = 'half'
        elif env_dtype in ('fp32', 'float32'):
            selected_dtype = 'float32'
        else:
            if cc is None:
                selected_dtype = 'float32' if not torch.cuda.is_available() else 'half'
            else:
                selected_dtype = 'bfloat16' if cc >= 8.0 else 'half'

        if selected_dtype == 'bfloat16' and (cc is None or cc < 8.0):
            print(f"[WARN] Requested bfloat16 but GPU compute capability {cc or 'unknown'} < 8.0; falling back to half.")
            selected_dtype = 'half'

        print(f"[DEBUG] Selected dtype for vLLM: {selected_dtype}")

        def _to_int(val, default):
            try:
                v = int(val)
                return v if v > 0 else default
            except Exception:
                return default

        def _to_float(val, default, lo=0.5, hi=0.98):
            try:
                v = float(val)
                if v < lo:
                    v = lo
                if v > hi:
                    v = hi
                return v
            except Exception:
                return default

        max_model_len = _to_int(os.environ.get('MAX_MODEL_LEN', ''), 2048)
        max_concurrency = _to_int(os.environ.get('MAX_CONCURRENCY', ''), 1)
        gpu_mem_util = _to_float(os.environ.get('GPU_MEMORY_UTILIZATION', ''), 0.95)

        print(f"[DEBUG] vLLM memory knobs: max_model_len={max_model_len}, max_num_seqs={max_concurrency}, gpu_memory_utilization={gpu_mem_util}")

        enforce_eager_env = os.environ.get('ENFORCE_EAGER', '').strip().lower()
        enforce_eager_param = enforce_eager_env in ('1', 'true', 'yes', 'on')

        env_block_size = os.environ.get('BLOCK_SIZE', '').strip()
        allowed_blocks = (16, 32, 64) if (cc is None or cc < 8.0) else (16, 32, 64, 128, 256)
        default_bs = 16
        try:
            bs_val = int(env_block_size) if env_block_size else default_bs
        except Exception:
            bs_val = default_bs
        if bs_val not in allowed_blocks:
            bs_val = default_bs
        selected_block_size = bs_val

        llm = LLM(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=selected_block_size,
            enforce_eager=enforce_eager_param,
            trust_remote_code=True,
            max_model_len=max_model_len,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem_util,
            disable_mm_preprocessor_cache=True,
            dtype=selected_dtype
        )
        
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
        
        max_tokens_env = os.environ.get('MAX_TOKENS', '').strip()
        try:
            max_tokens = int(max_tokens_env) if max_tokens_env else 2048
        except Exception:
            max_tokens = 2048

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        
        print("Model initialization complete!")

def pdf_to_images_high_quality(pdf_data: bytes, dpi: int = 144) -> List[Image.Image]:
    """Convert PDF bytes to high-quality PIL Images"""
    images = []
    dpi_env = os.environ.get('PDF_DPI', '').strip()
    effective_dpi = dpi
    try:
        if dpi_env:
            dpi_val = int(dpi_env)
            if dpi_val > 0 and dpi_val <= 300:
                effective_dpi = dpi_val
    except Exception:
        pass
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name
    
    try:
        pdf_document = fitz.open(temp_pdf_path)
        zoom = effective_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
    finally:
        os.unlink(temp_pdf_path)
    
    return images

def process_single_image(image: Image.Image, prompt: str = PROMPT) -> str:
    """Process a single image with DeepSeek-OCR using the specified prompt"""
    request_item = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                prompt=prompt,
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        }
    }
    
    outputs = llm.generate([request_item], sampling_params=sampling_params)
    result = outputs[0].outputs[0].text
    
    if '<｜end▁of▁sentence｜>' in result:
        result = result.replace('<｜end▁of▁sentence｜>', '')
    
    return result

async def process_job_async(job_id: str, prompt: str):
    """Process a job asynchronously in the background"""
    try:
        # Get job
        job = await job_manager.get_job(job_id)
        if not job:
            return
        
        # Mark as started
        await job_manager.start_job(job_id)
        
        # Load PDF
        input_path = job_manager.get_job_file_path(job_id, "input")
        if not input_path or not input_path.exists():
            await job_manager.fail_job(job_id, "Input file not found")
            return
        
        with open(input_path, 'rb') as f:
            pdf_data = f.read()
        
        # Convert to images
        images = pdf_to_images_high_quality(pdf_data)
        
        # Limit pages if configured
        max_pages_env = os.environ.get('MAX_PAGES', '').strip()
        try:
            max_pages = int(max_pages_env) if max_pages_env else 0
        except Exception:
            max_pages = 0
        if max_pages > 0 and len(images) > max_pages:
            images = images[:max_pages]
        
        if not images:
            await job_manager.fail_job(job_id, "No images extracted from PDF")
            return
        
        # Process each page
        results = []
        for page_num, image in enumerate(images):
            try:
                result = process_single_image(image, prompt)
                results.append(result)
                
                # Update progress
                await job_manager.update_job_progress(job_id, page_num + 1)
                
            except Exception as e:
                print(f"[ERROR] Page {page_num + 1} failed: {str(e)}")
                results.append(f"[Error processing page {page_num + 1}: {str(e)}]")
        
        # Combine results
        markdown_text = "\n\n<--- Page Split --->\n\n".join(results)
        
        # Create metadata
        metadata = {
            "filename": job.filename,
            "total_pages": len(images),
            "processed_at": datetime.now().isoformat(),
            "processing_time_seconds": time.time() - job.started_at,
            "prompt_used": prompt
        }
        
        # Mark as complete
        await job_manager.complete_job(job_id, markdown_text, metadata)
        
    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {str(e)}")
        await job_manager.fail_job(job_id, str(e))

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    initialize_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DeepSeek-OCR Async API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "job_storage_path": str(JOB_STORAGE_DIR),
        "active_jobs": len([j for j in job_manager.jobs.values() if j.status == JobStatus.PROCESSING]),
        "queued_jobs": len([j for j in job_manager.jobs.values() if j.status == JobStatus.QUEUED]),
        "total_jobs": len(job_manager.jobs)
    }

@app.post("/jobs/create", response_model=JobCreateResponse)
async def create_job_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Create a new OCR job"""
    try:
        # Read PDF data
        pdf_data = await file.read()
        
        # Get page count
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name
        
        try:
            doc = fitz.open(temp_pdf_path)
            total_pages = doc.page_count
            doc.close()
        finally:
            os.unlink(temp_pdf_path)
        
        # Create job
        job_id = await job_manager.create_job(file.filename, pdf_data, total_pages)
        
        # Start processing in background
        use_prompt = prompt if prompt else PROMPT
        background_tasks.add_task(process_job_async, job_id, use_prompt)
        
        return JobCreateResponse(
            success=True,
            job_id=job_id,
            message=f"Job created successfully. Total pages: {total_pages}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status and progress"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,
        progress=job.progress,
        total_pages=job.total_pages,
        processed_pages=job.processed_pages,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error
    )

@app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the completed job result"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status.value}")
    
    result_path = job_manager.get_job_file_path(job_id, "result")
    if not result_path or not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=result_path,
        media_type="text/markdown",
        filename=f"{job.filename}.md"
    )

@app.get("/jobs/{job_id}/metadata")
async def get_job_metadata(job_id: str):
    """Get job result metadata"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status.value}")
    
    metadata_path = job_manager.get_job_file_path(job_id, "metadata")
    if not metadata_path or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return JSONResponse(content=metadata)

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    await job_manager.cancel_job(job_id)
    return {"success": True, "message": "Job cancelled"}

@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(limit: int = 100):
    """List recent jobs"""
    jobs = await job_manager.list_jobs(limit)
    
    return JobListResponse(
        jobs=[
            JobResponse(
                job_id=job.job_id,
                filename=job.filename,
                status=job.status.value,
                progress=job.progress,
                total_pages=job.total_pages,
                processed_pages=job.processed_pages,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error
            )
            for job in jobs
        ]
    )

if __name__ == "__main__":
    print("Starting DeepSeek-OCR Async API server...")
    uvicorn.run(
        "start_server_async:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
