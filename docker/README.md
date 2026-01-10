# Docker Files

This folder contains files that are copied into the Docker container.

## Structure

```
docker/
??? start_server.py              # FastAPI server (main entry point)
??? overrides/                   # Files that override DeepSeek-OCR originals
    ??? config.py                # ? DeepSeek-OCR-vllm/config.py
    ??? deepseek_ocr.py          # ? DeepSeek-OCR-vllm/deepseek_ocr.py
    ??? run_dpsk_ocr_pdf.py      # ? DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py
    ??? run_dpsk_ocr_image.py    # ? DeepSeek-OCR-vllm/run_dpsk_ocr_image.py
    ??? run_dpsk_ocr_eval_batch.py # ? DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py
    ??? process/
        ??? image_process.py     # ? DeepSeek-OCR-vllm/process/image_process.py
```

## Files

### `start_server.py`
The FastAPI server that exposes the OCR API endpoints:
- `GET /health` - Health check and availability status
- `POST /jobs/create` - Create a new OCR job
- `GET /jobs/{job_id}` - Get job status and progress
- `GET /jobs/{job_id}/download` - Download completed result

### `overrides/`
These files replace the original DeepSeek-OCR files to add customizations:
- Memory optimizations for 8GB GPUs
- Enhanced error handling
- API-friendly interfaces

## Building

```bash
docker-compose build
docker-compose up -d
```

## API Usage

```bash
# Check health
curl http://localhost:8000/health

# Create job
curl -X POST http://localhost:8000/jobs/create \
  -F "file=@document.pdf" \
  -F "prompt=<image>Convert to markdown."

# Check status
curl http://localhost:8000/jobs/{job_id}

# Download result
curl http://localhost:8000/jobs/{job_id}/download
```
