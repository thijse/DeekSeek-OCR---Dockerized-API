# DeepSeek-OCR Samples

Minimal example scripts demonstrating how to use the DeepSeek-OCR library.

## Prerequisites

1. Docker server running: `docker-compose up -d`
2. Virtual environment activated: `.\venv\Scripts\Activate.ps1`

## Samples

### Basic PDF to Markdown
```bash
python samples/sample_pdf_to_markdown.py input.pdf
```
Converts a PDF to Markdown format with post-processing (tag cleanup, page split removal).

### Plain Text Extraction (OCR)
```bash
python samples/sample_pdf_to_ocr.py input.pdf
```
Extracts plain text from a PDF.

### Custom Prompt
```bash
python samples/sample_custom_prompt.py input.pdf
python samples/sample_custom_prompt.py input.pdf custom_prompt.yaml
```
Uses a custom prompt from a YAML file. Returns raw output (no post-processing).

### With Image Extraction
```bash
python samples/sample_with_images.py input.pdf
```
Converts PDF to Markdown and extracts detected images to a ZIP file.

### Batch Processing
```bash
python samples/sample_batch_process.py data/
```
Processes all PDFs in a folder sequentially.

## Library Usage

All samples use the `Lib` module:

```python
from Lib import get_config, OCRClient, PostProcessor, FileManager

# Initialize
config = get_config()
client = OCRClient(config)
postprocessor = PostProcessor(config)

# Create job
job_id, error = client.create_job("input.pdf", "<image>Convert to markdown.")

# Poll for status
status, error = client.get_job_status(job_id)

# Download result
result, error = client.download_result(job_id)

# Post-process
cleaned, images, zip_path = postprocessor.process(result, pdf_path="input.pdf")
```

## Full CLI Tool

For more options, use the CLI tool:
```bash
python pdf_to_markdown_cli.py --help
```
