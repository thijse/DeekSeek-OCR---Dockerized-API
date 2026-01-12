# DeepSeek-OCR vLLM Docker Deployment

This Docker setup provides a complete DeepSeek-OCR service with vLLM backend

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA 11.8+ support
- **GPU Memory**: Minimum 16GB VRAM (recommended: 40GB+ A100)
- **System RAM**: Minimum 32GB (recommended: 64GB+)
- **Storage**: 50GB+ free space for model and containers

### Software Requirements
- **Docker** 20.10+ with GPU support
- **Docker Compose** 2.0+
- **NVIDIA Container Toolkit** installed
- **CUDA 11.8** compatible drivers

## Quick Start

### 1. Prepare Model Weights

Create a directory for model weights and download the DeepSeek-OCR model:

```bash
mkdir -p models
# Option 1: Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR

# Option 2: Using git
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR models/deepseek-ai/DeepSeek-OCR
```

### 2. Build and Run

### Windows Users

```cmd
REM Build the Docker image
build.bat

REM Start the service
docker-compose up -d

REM Check logs
docker-compose logs -f deepseek-ocr
```

### Linux/macOS Users

```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f deepseek-ocr
```

### 3. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Should return something like:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/deepseek-ai/DeepSeek-OCR",
  "cuda_available": true,
  "cuda_device_count": 1
}
```

## API Usage

### Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Process Single Image
```bash
curl -X POST "http://localhost:8000/ocr/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

#### Process PDF
```bash
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

#### Batch Processing
```bash
curl -X POST "http://localhost:8000/ocr/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@document.pdf" \
  -F "files=@image2.png"
```

### Response Format

#### Single Image Response
```json
{
  "success": true,
  "result": "# Document Title\n\nThis is the OCR result in markdown format...",
  "page_count": 1
}
```

#### PDF Response
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "result": "# Page 1 Content\n...",
      "page_count": 1
    },
    {
      "success": true,
      "result": "# Page 2 Content\n...",
      "page_count": 2
    }
  ],
  "total_pages": 2,
  "filename": "document.pdf"
}
```

## Configuration

### Environment Variables

Edit `docker-compose.yml` to adjust these settings:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0                    # GPU device to use
  - MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR  # Model path
  - MAX_CONCURRENCY=50                         # Max concurrent requests
  - GPU_MEMORY_UTILIZATION=0.85                # GPU memory usage (0.1-1.0)
```

### Performance Tuning

#### For High-Throughput Processing
```yaml
environment:
  - MAX_CONCURRENCY=100
  - GPU_MEMORY_UTILIZATION=0.95
```

#### For Memory-Constrained Systems
```yaml
environment:
  - MAX_CONCURRENCY=10
  - GPU_MEMORY_UTILIZATION=0.7
```

## Advanced Usage

### Custom API Integration

#### Python Client
```python
import requests

class DeepSeekOCRClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def process_image(self, image_path):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/ocr/image",
                files={"file": f}
            )
        return response.json()
    
    def process_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/ocr/pdf",
                files={"file": f}
            )
        return response.json()

# Usage
client = DeepSeekOCRClient()
result = client.process_image("document.jpg")
print(result["result"])
```

#### JavaScript Client
```javascript
class DeepSeekOCR {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async processImage(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.baseUrl}/ocr/image`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async processPDF(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.baseUrl}/ocr/pdf`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
}

// Usage in browser
const ocr = new DeepSeekOCR();
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const result = await ocr.processImage(file);
    console.log(result.result);
});
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Reduce concurrency and GPU memory usage
# Edit docker-compose.yml:
environment:
  - MAX_CONCURRENCY=10
  - GPU_MEMORY_UTILIZATION=0.7
```

#### 2. Model Loading Issues
```bash
# Check model directory structure
ls -la models/deepseek-ai/DeepSeek-OCR/

# Verify model files are present
docker-compose exec deepseek-ocr ls -la /app/models/deepseek-ai/DeepSeek-OCR/
```

#### 3. CUDA Errors
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### 4. Slow Performance
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check container logs
docker-compose logs -f deepseek-ocr
```

### Debug Mode

For debugging, you can run the container with additional tools:

```bash
# Run with shell access
docker-compose run --rm deepseek-ocr bash

# Check model loading
python -c "
import sys
sys.path.insert(0, '/app/DeepSeek-OCR-master/DeepSeek-OCR-vllm')
from config import MODEL_PATH
print(f'Model path: {MODEL_PATH}')
print(f'Model exists: {os.path.exists(MODEL_PATH)}')
"
```

### Monitoring

```bash
# Add monitoring to docker-compose.yml
services:
  deepseek-ocr:
    # ... existing config
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

### Scaling

For multiple GPU instances:

```yaml
services:
  deepseek-ocr-1:
    extends:
      file: docker-compose.yml
      service: deepseek-ocr
    environment:
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
  
  deepseek-ocr-2:
    extends:
      file: docker-compose.yml
      service: deepseek-ocr
    environment:
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "8001:8000"
```

## Support

For issues related to:
- **Docker setup**: Check this README first
- **DeepSeek-OCR model**: Refer to the [official repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- **vLLM**: Refer to [vLLM documentation](https://docs.vllm.ai/)

## Configurations

This project’s server dynamically adapts to the GPU class via configuration. No code changes are required to switch between older pre-Ampere GPUs and newer Ampere/Ada/Hopper GPUs. The selection logic and tunables are controlled through environment variables and auto-detection in the server.

Key runtime knobs (set via docker-compose environment):
- DTYPE: precision for model weights and compute
- BLOCK_SIZE: paged KV-cache block size used by vLLM’s attention kernels
- ENFORCE_EAGER: toggle CUDA graph capture vs eager execution
- MAX_MODEL_LEN: maximum tokens per request (affects KV cache size)
- MAX_CONCURRENCY: concurrent sequences (affects KV cache size)
- GPU_MEMORY_UTILIZATION: fraction of total VRAM vLLM can use

Where these are applied in code:
- Precision selection and auto-detect: [`LLM()`](start_server.py:86)
- Memory knobs: [`LLM()`](start_server.py:138)
- Block size selection (allowed values vary by compute capability): [`LLM()`](start_server.py:144)
- Eager vs CUDA graph capture toggle: [`LLM()`](start_server.py:156)

Compose environment example (edit your compose file):
- File location: [`docker-compose.yml`](docker-compose.yml)
- Example entries (with comments):
  - CUDA_VISIBLE_DEVICES=0
  - MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR
  - MAX_CONCURRENCY=1
  - MAX_MODEL_LEN=2048
  - GPU_MEMORY_UTILIZATION=0.95
  - DTYPE=${DTYPE:-}
  - BLOCK_SIZE=16
  - ENFORCE_EAGER=0

GPU classes and recommended configuration

Use your GPU’s compute capability (CC) to decide the settings:
- CC < 8.0 = pre-Ampere GPUs (Turing/Volta):
  - Examples: GeForce RTX 2060/2070/2080/Titan RTX (Turing, CC 7.5), Tesla T4 (Turing, CC 7.5), Tesla V100 (Volta, CC 7.0)
  - Recommended config:
    - DTYPE=half
      - Reason: bfloat16 is unsupported on CC < 8.0. The server auto-clamps to “half” if you set bfloat16 accidentally.
    - BLOCK_SIZE=16 (supported: 16, 32, 64 only)
      - Reason: xFormers backend on Turing/Volta does not accept larger page sizes.
    - ENFORCE_EAGER=1 (optional)
      - Use if CUDA graph capture fails due to memory constraints or model dynamism.
    - MAX_MODEL_LEN=2048, MAX_CONCURRENCY=1
      - For 8 GB VRAM GPUs, this keeps KV cache within remaining memory alongside ~6.23 GiB model weights.
    - GPU_MEMORY_UTILIZATION=0.95
      - Allow vLLM to use most of VRAM. Lower slightly if you see runtime OOM.

- CC ≥ 8.0 = Ampere/Ada/Hopper GPUs:
  - Examples: GeForce RTX 30xx (Ampere, CC 8.6), RTX 40xx (Ada, CC 8.9), A100 (Ampere, CC 8.0), H100 (Hopper, CC 9.0)
  - Recommended config:
    - DTYPE=bfloat16
      - Reason: bfloat16 widely supported on CC ≥ 8.0; helps accuracy and performance for many models.
    - BLOCK_SIZE=128 or 256 (supported: 16, 32, 64, 128, 256)
      - Larger pages can improve throughput. If you encounter kernel errors, fall back to 64.
    - ENFORCE_EAGER=0
      - Prefer CUDA graph capture for better perf unless you see capture-related errors; toggle via env if needed.
    - MAX_MODEL_LEN=4096–8192 (adjust based on VRAM), MAX_CONCURRENCY≥1
      - Increase gradually; KV cache scales heavily with sequence length and concurrency.
    - GPU_MEMORY_UTILIZATION=0.95–0.98
      - Higher values can be used on large VRAM GPUs, but monitor for OOM.

Switching between GPU classes (configuration-only steps)

1) Identify your GPU compute capability:
   - NVIDIA reference: search “CUDA GPUs compute capability” online, or use nvidia-smi queries.
   - Examples:
     - Turing: CC 7.5 (RTX 20xx, T4)
     - Volta: CC 7.0 (V100)
     - Ampere: CC 8.0–8.6 (RTX 30xx, A100)
     - Ada: CC 8.9 (RTX 40xx)
     - Hopper: CC 9.0 (H100)

2) Edit your compose environment:
   - For pre-Ampere (CC < 8.0):
     - Set DTYPE=half
     - Set BLOCK_SIZE=16
     - Optionally set ENFORCE_EAGER=1
     - Keep conservative memory knobs (MAX_MODEL_LEN=2048, MAX_CONCURRENCY=1, GPU_MEMORY_UTILIZATION=0.95)
   - For Ampere/Ada/Hopper (CC ≥ 8.0):
     - Set DTYPE=bfloat16
     - Set BLOCK_SIZE=128 or 256
     - Set ENFORCE_EAGER=0
     - Increase MAX_MODEL_LEN and MAX_CONCURRENCY according to VRAM; adjust GPU_MEMORY_UTILIZATION to 0.95–0.98

3) Rebuild and restart:
   - docker-compose up -d --build
   - Verify health: curl http://localhost:8000/health should report "model_loaded": true

Notes and caveats
- The server auto-detects compute capability to validate allowed BLOCK_SIZE values at runtime and will fallback safely if you pick an unsupported size.
- On WSL, pin_memory is disabled automatically; expect slightly lower throughput.
- FlashAttention-2 is unavailable on Turing/Volta; the stack will use xFormers on those GPUs. Ampere/Ada/Hopper can leverage FlashAttention-2 automatically with the provided container dependencies.
- If you encounter cudagraph capture issues, set ENFORCE_EAGER=1. This is read at runtime by [`LLM()`](start_server.py:156) and does not require a code change.

Quick reference for compose (older vs newer GPUs)
- Pre-Ampere (Turing/Volta, CC < 8.0):
  - DTYPE=half
  - BLOCK_SIZE=16
  - ENFORCE_EAGER=1 (optional)
  - MAX_MODEL_LEN=2048, MAX_CONCURRENCY=1, GPU_MEMORY_UTILIZATION=0.95
- Ampere/Ada/Hopper (CC ≥ 8.0):
  - DTYPE=bfloat16
  - BLOCK_SIZE=128 (or 256)
  - ENFORCE_EAGER=0
  - MAX_MODEL_LEN 4096–8192 (as VRAM allows), MAX_CONCURRENCY≥1, GPU_MEMORY_UTILIZATION 0.95–0.98

All of the above are configuration-only. The code path already supports both classes via environment variables and auto-detection—no additional edits are required beyond changing the compose environment entries in [`docker-compose.yml`](docker-compose.yml:10).
