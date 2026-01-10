# Docker Files

This folder contains files that are copied into the Docker container.

## Structure

```
docker/
??? start_server.py                    # FastAPI server (main entry point)
??? entrypoint.py                      # Container entrypoint (starts API + optional GUI)
??? config_location.py                 # Docker-specific configuration
??? README.md                          # This file
??? overrides/                         # Files that override DeepSeek-OCR originals
    ??? config.py                      # ? DeepSeek-OCR-vllm/config.py
    ??? deepseek_ocr.py                # ? DeepSeek-OCR-vllm/deepseek_ocr.py
    ??? run_dpsk_ocr_pdf.py            # ? DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py
    ??? run_dpsk_ocr_image.py          # ? DeepSeek-OCR-vllm/run_dpsk_ocr_image.py
    ??? run_dpsk_ocr_eval_batch.py     # ? DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py
    ??? process/
        ??? image_process.py           # ? DeepSeek-OCR-vllm/process/image_process.py
```

## Files

### `start_server.py`
The FastAPI server that exposes the OCR API endpoints:
- `GET /health` - Health check and availability status
- `POST /jobs/create` - Create a new OCR job
- `GET /jobs/{job_id}` - Get job status and progress
- `GET /jobs/{job_id}/download` - Download completed result
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

### `entrypoint.py`
Python entrypoint script that:
- Starts the API server on port 8000
- Optionally starts the GUI on port 7863 (if `ENABLE_GUI=true`)

### `config_location.py`
Docker-specific configuration that gets copied to `/app/Lib/config_location.py`:
- Sets paths to `/app/data/uploads`, `/app/data/results`, etc.
- Sets GUI port to 7863 (different from local 7862)
- Marks environment as "docker"

### `overrides/`
These files replace the original DeepSeek-OCR files to add customizations:
- Memory optimizations for 8GB GPUs
- Enhanced error handling
- API-friendly interfaces
- Fixed prompt parameter handling

## Building

```bash
# Build the Docker image
docker-compose build

# Start API only
docker-compose up -d

# Start with GUI enabled
ENABLE_GUI=true docker-compose up -d
```

## Ports

| Port | Service |
|------|---------|
| 8000 | API Server |
| 7863 | GUI (when enabled) |

## API Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage Examples

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
curl http://localhost:8000/jobs/{job_id}/download -o result.md
```

## Environment Variables

Set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/deepseek-ai/DeepSeek-OCR` | Path to model weights |
| `MAX_CONCURRENCY` | `1` | Max concurrent sequences |
| `MAX_MODEL_LEN` | `2048` | Max sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory usage (0-1) |
| `DTYPE` | auto | Data type: `half`, `bfloat16`, `float32` |
| `BLOCK_SIZE` | `16` | KV cache block size |
| `MAX_TOKENS` | `1024` | Max output tokens per page |
| `PDF_DPI` | `96` | PDF rendering DPI |
| `MAX_PAGES` | `0` | Max pages (0 = unlimited) |
| `ENABLE_GUI` | `false` | Enable GUI on port 7863 |
