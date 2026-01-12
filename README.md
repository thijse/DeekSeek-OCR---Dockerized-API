![](http://Docs/DeepSeek-OCR.png)
# DeepSeek-OCR: PDF to Markdown Converter

A DeepSeek-OCR-based solution that converts PDF documents to Markdown format. This project provides a REST API, Web GUI, and CLI tool.

## Features

- **REST API** - Async job-based processing with progress tracking
- **Web GUI** - Gradio interface for easy file uploads and queue management
- **CLI Tool** - Command-line interface for batch processing
- **Docker** - Fully containerized with optional GUI
- **Modest GPU Support** - works for older GPUs (RTX 2070, etc)

---

## Quick Start

### 1. Download Model Weights

```bash
# Create models directory
mkdir -p models

# Download using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR
```

### 2. Build and Start Docker

```bash
# Build the Docker image
docker-compose build

# Start the API server
docker-compose up -d

# Wait for model to load (~60 seconds), then verify
curl http://localhost:8000/health
```

### 3. Use the Service

**Option A: Web GUI (running inside of Docker)**
```bash
# Start with GUI enabled inside Docker
Set Enable GUI to true in docker-compose.yml (default)
docker-compose up -d

# Open http://localhost:7863
```

**Option B: Web GUI (Local)**
```bash
# Install dependencies
pip install -r requirements.txt

# Start local GUI (connects to Docker API)
python GUI.py
# Open http://localhost:7862
```

**Option C: CLI Tool**
```bash
python pdf_to_markdown_cli.py document.pdf -o output/
```

---

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support
- **GPU Memory**: Minimum 8GB VRAM (tested on RTX 2070)
- **System RAM**: Minimum 16GB
- **Storage**: 20GB+ for model weights

### Software Requirements
- **Python 3.8+**
- **Docker** 20.10+ with GPU support
- **Docker Compose** 2.0+
- **NVIDIA Container Toolkit**

---

## URLs & Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| **API Health** | http://localhost:8000/health | Server status and availability |
| **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **API Docs (ReDoc)** | http://localhost:8000/redoc | Alternative API documentation |
| **Local GUI** | http://localhost:7862 | Gradio interface (local Python) |
| **Docker GUI** | http://localhost:7863 | Gradio interface (inside Docker) |

---

## Web GUI

The Gradio web interface provides:
- Drag & drop PDF upload
- File queue management
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Real-time progress tracking
- Image extraction options
- Result preview and download

### Local GUI (Recommended for Development)
```bash
# Requires Docker API running on localhost:8000
python GUI.py
```

### Docker GUI
```bash
# Or set in docker-compose.yml:
# environment:
#   - ENABLE_GUI=true
```

---

## CLI Tool

The CLI tool (`pdf_to_markdown_cli.py`) supports all processing options:

```bash
# Basic usage
python pdf_to_markdown_cli.py input.pdf

# Specify output directory
python pdf_to_markdown_cli.py input.pdf -o output/

# Process all PDFs in a folder
python pdf_to_markdown_cli.py folder/ --batch

# Process files from a list
python pdf_to_markdown_cli.py --list files.txt

# OCR mode (plain text extraction)
python pdf_to_markdown_cli.py input.pdf --mode ocr

# Custom prompt from YAML file
python pdf_to_markdown_cli.py input.pdf --prompt custom_prompt.yaml

# Extract images to ZIP
python pdf_to_markdown_cli.py input.pdf --extract-images

# Keep page split markers
python pdf_to_markdown_cli.py input.pdf --keep-page-splits

# Raw output (no post-processing)
python pdf_to_markdown_cli.py input.pdf --no-clean

# Verbose output
python pdf_to_markdown_cli.py input.pdf -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Output directory |
| `--batch, -b` | Process all PDFs in folder |
| `--list, -l` | File containing list of PDFs |
| `--mode, -m` | Processing mode: `markdown` or `ocr` |
| `--prompt, -p` | Custom prompt YAML file |
| `--extract-images, -i` | Extract images to ZIP |
| `--keep-page-splits` | Keep page split markers |
| `--no-clean` | Skip post-processing |
| `--suffix, -s` | Output filename suffix |
| `--host` | API host (default: localhost) |
| `--port` | API port (default: 8000) |
| `--verbose, -v` | Verbose output |

---

## REST API

The API uses an async job-based workflow:

### Workflow
1. **Check availability**: `GET /health`
2. **Submit job**: `POST /jobs/create`
3. **Poll status**: `GET /jobs/{job_id}`
4. **Download result**: `GET /jobs/{job_id}/download`

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "available": true,
  "model_loaded": true,
  "current_job": null
}
```

#### Create Job
```bash
curl -X POST http://localhost:8000/jobs/create \
  -F "file=@document.pdf" \
  -F "prompt=<image>Convert the content to markdown."
```

Response:
```json
{
  "success": true,
  "job_id": "20260110_123456_abc12345",
  "message": "Job started. Total pages: 10"
}
```

#### Get Job Status
```bash
curl http://localhost:8000/jobs/{job_id}
```

Response:
```json
{
  "job_id": "20260110_123456_abc12345",
  "filename": "document.pdf",
  "status": "processing",
  "progress": 45.0,
  "total_pages": 10,
  "processed_pages": 4
}
```

#### Download Result
```bash
curl http://localhost:8000/jobs/{job_id}/download -o result.md
```

### Default Prompts

| Mode | Prompt |
|------|--------|
| Markdown | `<image>Convert the content of the image to markdown.` |
| OCR | `<image>Extract all text from the image.` |

---

## 📁 Sample Scripts

The `samples/` directory contains minimal example scripts:

| Script | Description |
|--------|-------------|
| `sample_pdf_to_markdown.py` | Basic PDF to Markdown conversion |
| `sample_pdf_to_ocr.py` | Plain text extraction |
| `sample_custom_prompt.py` | Using custom prompts from YAML |
| `sample_with_images.py` | Image extraction with ZIP output |
| `sample_batch_process.py` | Process multiple PDFs from a folder |

```bash
# Run a sample
python samples/sample_pdf_to_markdown.py document.pdf
```

---

## ⚙️ Configuration

### Docker Environment Variables

Edit `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR
  - MAX_CONCURRENCY=1          # Keep at 1 for 8GB GPUs
  - MAX_MODEL_LEN=2048         # Reduce for memory savings
  - GPU_MEMORY_UTILIZATION=0.95
  - DTYPE=half                 # half, bfloat16, or float32
  - BLOCK_SIZE=16              # KV cache block size
  - MAX_TOKENS=1024            # Max output tokens per page
  - PDF_DPI=96                 # PDF rendering DPI
  - MAX_PAGES=0                # 0 = unlimited
  - ENABLE_GUI=false           # Set to true for Docker GUI
```

### Custom Prompts

Create `custom_prompt.yaml`:
```yaml
prompt: '<image>Extract all tables and format as CSV.'
```

Use with CLI:
```bash
python pdf_to_markdown_cli.py input.pdf --prompt custom_prompt.yaml
```

---

## Project Structure

```
DeepSeek-OCR/
├── GUI.py                     # Gradio web interface
├── pdf_to_markdown_cli.py     # CLI tool
├── Lib/                       # Shared Python library
│   ├── config.py              # Configuration class
│   ├── config_location.py     # Local environment settings
│   ├── ocr_client.py          # API client
│   ├── postprocessor.py       # Output post-processing
│   └── file_utils.py          # File management utilities
├── docker/                    # Docker-related files
│   ├── start_server.py        # FastAPI server
│   ├── entrypoint.py          # Container entrypoint
│   ├── config_location.py     # Docker environment settings
│   └── overrides/             # Files that override DeepSeek-OCR
│       ├── config.py
│       ├── deepseek_ocr.py
│       └── ...
├── samples/                   # Example scripts
│   ├── sample_pdf_to_markdown.py
│   ├── sample_pdf_to_ocr.py
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── data/                      # Input/output directory
│   ├── uploads/               # Uploaded files
│   ├── results/               # Processing results
│   └── images/                # Extracted images
└── models/                    # Model weights (download separately)
    └── deepseek-ai/
        └── DeepSeek-OCR/
```

---

## Troubleshooting

### Out of Memory Errors
```yaml
# Reduce memory usage in docker-compose.yml:
environment:
  - MAX_CONCURRENCY=1
  - MAX_MODEL_LEN=1024
  - GPU_MEMORY_UTILIZATION=0.90
```

### Model Loading Issues
```bash
# Check model directory
ls -la models/deepseek-ai/DeepSeek-OCR/

# Check container logs
docker-compose logs -f deepseek-ocr
```

### API Connection Errors
```bash
# Check if API is running
curl http://localhost:8000/health

# Restart the service
docker-compose restart
```

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

---

## License

This project follows the same license as DeepSeek-OCR. See the [original repository](https://github.com/deepseek-ai/DeepSeek-OCR) for details.

---

## Support

- **Docker/API issues**: Check this README
- **DeepSeek-OCR model**: [Official repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- **vLLM**: [vLLM documentation](https://docs.vllm.ai/)
