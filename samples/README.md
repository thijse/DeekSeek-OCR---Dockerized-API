# DeepSeek-OCR Samples

Minimal example scripts demonstrating how to use the DeepSeek-OCR library.

## Prerequisites

1. Docker server running: `docker-compose up -d`
2. Wait for model to load (~60 seconds)
3. Virtual environment activated: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Linux/Mac)

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
from Lib import config, OCRClient, PostProcessor, FileManager

# Initialize components using the shared config
client = OCRClient(config)
postprocessor = PostProcessor(config)
file_manager = FileManager(config)

# Check API health
healthy, msg, available = client.check_health()

# Create job
job_id, error = client.create_job("input.pdf", "<image>Convert to markdown.")

# Poll for status
status, error = client.get_job_status(job_id)
# status['status'] can be: 'processing', 'completed', 'failed'
# status['progress'] is 0-100

# Download result
result, error = client.download_result(job_id)

# Post-process (clean tags, extract images)
cleaned, image_paths, zip_path = postprocessor.process(
    result,
    pdf_path="input.pdf",
    extract_images=True,
    remove_page_splits=True
)

# Save result
file_manager.save_result("input.pdf", cleaned, "markdown", prompt)
```

## Full CLI Tool

For more options, use the CLI tool:
```bash
python pdf_to_markdown_cli.py --help
```

### CLI Examples
```bash
# Basic conversion
python pdf_to_markdown_cli.py input.pdf

# Specify output directory
python pdf_to_markdown_cli.py input.pdf -o output/

# Batch process folder
python pdf_to_markdown_cli.py folder/ --batch

# OCR mode (plain text)
python pdf_to_markdown_cli.py input.pdf --mode ocr

# Custom prompt
python pdf_to_markdown_cli.py input.pdf --prompt custom_prompt.yaml

# Extract images
python pdf_to_markdown_cli.py input.pdf --extract-images

# Verbose output
python pdf_to_markdown_cli.py input.pdf -v
```

## Output Locations

| Type | Location |
|------|----------|
| Results | `data/results/` |
| Uploads | `data/uploads/` |
| Images | `data/images/` |

## Configuration

The samples use `Lib/config_location.py` for environment-specific settings:
- **Local**: Uses `data/uploads`, `data/results`, port 7862
- **Docker**: Uses `/app/data/uploads`, `/app/data/results`, port 7863
