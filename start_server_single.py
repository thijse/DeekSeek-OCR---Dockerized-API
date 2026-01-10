#!/usr/bin/env python3
"""
DeepSeek-OCR vLLM Server - Simplified Single-Job Mode
FastAPI wrapper for DeepSeek-OCR with vLLM backend.
Processes ONE job at a time. Client manages the queue.
"""

import os
import sys
import asyncio
import io
import tempfile
import uuid
import time
import shutil
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import fitz  # PyMuPDF
from PIL import Image

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


class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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


class SingleJobManager:
    """
    Simplified job manager - only ONE job at a time.
    No queue. Client is responsible for waiting and retrying.
    """
    
    def __init__(self):
        self.current_job: Optional[Job] = None
        self.last_completed_job: Optional[Job] = None
        self.lock = asyncio.Lock()
        self._clear_stale_jobs()
    
    def _clear_stale_jobs(self):
        """Clear any leftover job directories on startup"""
        try:
            for job_dir in JOB_STORAGE_DIR.iterdir():
                if job_dir.is_dir():
                    shutil.rmtree(job_dir, ignore_errors=True)
            print("[INFO] Cleared stale job directories")
        except Exception as e:
            print(f"[WARN] Error clearing stale jobs: {e}")
    
    def is_available(self) -> bool:
        """Check if server can accept a new job"""
        return self.current_job is None
    
    async def create_job(self, filename: str, pdf_data: bytes, total_pages: int):
        """Create a new job if available"""
        async with self.lock:
            if self.current_job is not None:
                return None, "Server busy - already processing a job"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            job_id = f"{timestamp}_{unique_id}"
            job_dir = JOB_STORAGE_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            input_path = job_dir / "input.pdf"
            with open(input_path, 'wb') as f:
                f.write(pdf_data)
            self.current_job = Job(
                job_id=job_id,
                filename=filename,
                status=JobStatus.PROCESSING,
                progress=0.0,
                total_pages=total_pages,
                processed_pages=0,
                created_at=time.time(),
                started_at=time.time()
            )
            return job_id, None
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        if self.current_job and self.current_job.job_id == job_id:
            return self.current_job
        if self.last_completed_job and self.last_completed_job.job_id == job_id:
            return self.last_completed_job
        return None
    
    async def update_progress(self, processed_pages: int):
        async with self.lock:
            if self.current_job:
                self.current_job.processed_pages = processed_pages
                if self.current_job.total_pages > 0:
                    self.current_job.progress = (processed_pages / self.current_job.total_pages) * 100.0
    
    async def complete_job(self, result_data: str):
        async with self.lock:
            if self.current_job:
                self.current_job.status = JobStatus.COMPLETED
                self.current_job.completed_at = time.time()
                self.current_job.progress = 100.0
                job_dir = JOB_STORAGE_DIR / self.current_job.job_id
                result_path = job_dir / "result.md"
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(result_data)
                if self.last_completed_job:
                    old_dir = JOB_STORAGE_DIR / self.last_completed_job.job_id
                    if old_dir.exists():
                        shutil.rmtree(old_dir, ignore_errors=True)
                self.last_completed_job = self.current_job
                self.current_job = None
    
    async def fail_job(self, error: str):
        async with self.lock:
            if self.current_job:
                self.current_job.status = JobStatus.FAILED
                self.current_job.completed_at = time.time()
                self.current_job.error = error
                job_dir = JOB_STORAGE_DIR / self.current_job.job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                self.last_completed_job = self.current_job
                self.current_job = None
    
    def get_input_path(self, job_id: str) -> Optional[Path]:
        path = JOB_STORAGE_DIR / job_id / "input.pdf"
        return path if path.exists() else None
    
    def get_result_path(self, job_id: str) -> Optional[Path]:
        path = JOB_STORAGE_DIR / job_id / "result.md"
        return path if path.exists() else None


app = FastAPI(
    title="DeepSeek-OCR API",
    description="Single-job OCR service. Client manages queue.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
sampling_params = None
job_manager = SingleJobManager()


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
    job_id: Optional[str] = None
    message: str


class HealthResponse(BaseModel):
    status: str
    available: bool
    model_loaded: bool
    current_job: Optional[JobResponse] = None


def initialize_model():
    global llm, sampling_params
    if llm is None:
        print("Initializing DeepSeek-OCR model...")
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
                return max(lo, min(hi, v))
            except Exception:
                return default
        max_model_len = _to_int(os.environ.get('MAX_MODEL_LEN', ''), 2048)
        max_concurrency = _to_int(os.environ.get('MAX_CONCURRENCY', ''), 1)
        gpu_mem_util = _to_float(os.environ.get('GPU_MEMORY_UTILIZATION', ''), 0.95)
        print(f"[DEBUG] vLLM: max_model_len={max_model_len}, max_num_seqs={max_concurrency}, gpu_memory_utilization={gpu_mem_util}")
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
        llm = LLM(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=bs_val,
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


def pdf_to_images(pdf_data: bytes, dpi: int = 144) -> List[Image.Image]:
    images = []
    dpi_env = os.environ.get('PDF_DPI', '').strip()
    try:
        if dpi_env:
            dpi_val = int(dpi_env)
            if 0 < dpi_val <= 300:
                dpi = dpi_val
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name
    try:
        pdf_document = fitz.open(temp_pdf_path)
        zoom = dpi / 72.0
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
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            return
        input_path = job_manager.get_input_path(job_id)
        if not input_path:
            await job_manager.fail_job("Input file not found")
            return
        with open(input_path, 'rb') as f:
            pdf_data = f.read()
        images = pdf_to_images(pdf_data)
        max_pages_env = os.environ.get('MAX_PAGES', '').strip()
        try:
            max_pages = int(max_pages_env) if max_pages_env else 0
        except Exception:
            max_pages = 0
        if max_pages > 0 and len(images) > max_pages:
            images = images[:max_pages]
        if not images:
            await job_manager.fail_job("No images extracted from PDF")
            return
        results = []
        for page_num, image in enumerate(images):
            try:
                print(f"[INFO] Processing page {page_num + 1}/{len(images)} for job {job_id}")
                result = process_single_image(image, prompt)
                results.append(result)
                await job_manager.update_progress(page_num + 1)
            except Exception as e:
                print(f"[ERROR] Page {page_num + 1} failed: {str(e)}")
                results.append(f"[Error processing page {page_num + 1}: {str(e)}]")
        markdown_text = "\n\n<--- Page Split --->\n\n".join(results)
        await job_manager.complete_job(markdown_text)
        print(f"[INFO] Job {job_id} completed successfully")
    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {str(e)}")
        await job_manager.fail_job(str(e))


@app.on_event("startup")
async def startup_event():
    initialize_model()


@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR API", "version": "3.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    current_job_response = None
    if job_manager.current_job:
        j = job_manager.current_job
        current_job_response = JobResponse(
            job_id=j.job_id,
            filename=j.filename,
            status=j.status.value,
            progress=j.progress,
            total_pages=j.total_pages,
            processed_pages=j.processed_pages,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
            error=j.error
        )
    return HealthResponse(
        status="healthy",
        available=job_manager.is_available(),
        model_loaded=llm is not None,
        current_job=current_job_response
    )


@app.post("/jobs/create", response_model=JobCreateResponse)
async def create_job_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    if not job_manager.is_available():
        raise HTTPException(status_code=503, detail="Server busy - already processing a job. Try again later.")
    try:
        pdf_data = await file.read()
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name
        try:
            doc = fitz.open(temp_pdf_path)
            total_pages = doc.page_count
            doc.close()
        finally:
            os.unlink(temp_pdf_path)
        job_id, error = await job_manager.create_job(file.filename, pdf_data, total_pages)
        if error:
            raise HTTPException(status_code=503, detail=error)
        use_prompt = prompt if prompt else PROMPT
        background_tasks.add_task(process_job_async, job_id, use_prompt)
        return JobCreateResponse(success=True, job_id=job_id, message=f"Job started. Total pages: {total_pages}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
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
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status.value}")
    result_path = job_manager.get_result_path(job_id)
    if not result_path:
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=result_path, media_type="text/markdown", filename=f"{job.filename}.md")


if __name__ == "__main__":
    print("Starting DeepSeek-OCR API server (Single-Job Mode)...")
    uvicorn.run("start_server_single:app", host="0.0.0.0", port=8000, reload=False, workers=1)
