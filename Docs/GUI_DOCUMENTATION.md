# DeepSeek-OCR PDF Converter - Application Documentation

## Overview

**GUI.py** is a Gradio web interface for converting PDF documents to Markdown or extracting text using the DeepSeek-OCR API. 

**Framework:** Gradio 3.50.2  
**Backend:** DeepSeek-OCR API (Docker)  
**Port:** 7862

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRADIO WEB CLIENT                            │
│                    (GUI.py - Port 7862)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • File Queue (client-side)                              │   │
│  │  • Sequential processing (one file at a time)            │   │
│  │  • Post-processing (tag cleanup, image extraction)       │   │
│  │  • Progress tracking via polling                         │   │
│  │  • Downloads: Markdown files + Image ZIPs                │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               DOCKER: DeepSeek-OCR Server                       │
│                (start_server.py - Port 8000)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Single-job mode (ONE job at a time)                   │   │
│  │  • Returns 503 if busy                                   │   │
│  │  • /health - availability + current job progress         │   │
│  │  • /jobs/create - submit PDF + prompt                    │   │
│  │  • /jobs/{id} - get status/progress                      │   │
│  │  • /jobs/{id}/download - get result                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---


## Technical Architecture

### Directory Structure
```
data/
├── uploads/     # Uploaded PDF files (timestamped)
├── results/     # Markdown output files + metadata JSON
└── images/      # Extracted image files
```

### Dependencies
```
gradio>=3.50.2
requests
PyMuPDF (fitz)
Pillow (PIL)
PyYAML
```

### Configuration
| Setting | Value | Description |
|---------|-------|-------------|
| TIMEOUT_SECONDS | 7200 | 2-hour timeout for large documents |
| ESTIMATED_SECONDS_PER_PAGE | 15 | Initial estimate (adapts over time) |
| API_BASE_URL | http://localhost:8000 | DeepSeek-OCR API endpoint |

---



## UI Components

### Left Panel (Controls)
1. **API Status** - Connection indicator with refresh button
2. **Drop Zone** - File upload area (auto-adds to queue)
3. **Queue Management** - Clear All / Clear Done buttons
4. **Queue Display** - Markdown list of queued files
5. **Settings**
   - Processing Mode (Markdown/OCR/Custom)
   - Extract Images checkbox
   - Remove Page Splits checkbox
   - Custom Prompt input (when Custom mode selected)
6. **Process/Cancel Buttons**
7. **Previous Results** - Dropdown to load saved results

### Right Panel (Output)
1. **Progress Status** - Single-line progress with ETA
2. **Result Tabs**
   - Rendered (scrollable markdown)
   - Raw (plain text)
3. **Downloads**
   - Markdown Results (multiple files)
   - Extracted Images ZIP

---

## Usage Guide

### Quick Start
1. **Start the API**: `docker-compose up -d`
2. **Launch GUI**: `python GUI_enhanced.py`
3. **Open Browser**: Navigate to http://127.0.0.1:7862
4. **Drop Files**: Drag PDFs onto the drop zone
5. **Configure**: Select mode and options
6. **Process**: Click "Process Queue"
7. **Download**: Get results from Downloads section

### Processing Modes

**Markdown Mode**
- Best for structured documents
- Preserves headings, lists, tables
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`

**OCR Mode**
- Best for raw text extraction
- Simpler output format
- Prompt: `<image>\nFree OCR.`

**Custom Prompt Mode**
- Full control over extraction
- Load from `custom_prompt.yaml` or type directly
- Must include `<image>` tag

### Post-Processing Options

**Extract Images** (default: off)
- Extracts images from coordinate tags
- Creates downloadable ZIP archive
- Replaces tags with markdown image links

**Remove Page Splits** (default: on)
- Removes `<--- Page Split --->` markers
- Creates seamless document flow

---

## Known Behaviors

1. **Progress at 100%**: When processing exceeds estimated time, progress holds at 100% and recalibrates the time/page estimate
2. **Cancel Timing**: Cancel takes effect between files, not mid-file
3. **Queue During Processing**: New files can be added but won't process until next run
4. **Cached Results**: Previously processed files load from cache instantly

---

## File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Uploads | `{name}_{timestamp}.pdf` | `report_20260107_120000.pdf` |
| Results | `{name}_{mode}.md` | `report_20260107_120000_MD.md` |
| Metadata | `{name}_{mode}_meta.json` | `report_20260107_120000_MD_meta.json` |
| Images | `{name}_img{n}_{timestamp}.jpg` | `report_img0_20260107_120000_123456.jpg` |
| Image ZIP | `{name}_images.zip` | `report_20260107_120000_images.zip` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API not connected | Run `docker-compose up -d` to start the API |
| Slow processing | Normal - ~15s/page, adapts over time |
| Incomplete tags in output | Fixed - truncated tags are now cleaned |
| Progress stuck at 95% | Fixed - now clamps at 100% with recalibration |
| Drop zone disappears | Fixed - zone persists after file drop |

---

## Credits

- **DeepSeek-OCR**: Underlying OCR model and API
- **Gradio**: Web interface framework
- **PyMuPDF**: PDF processing and image extraction
- **Pillow**: Image manipulation

---

*Documentation generated: January 7, 2026*
