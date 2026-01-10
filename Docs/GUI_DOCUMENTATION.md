# DeepSeek-OCR PDF Converter - Application Documentation

## Overview

**GUI.py** is a feature-rich Gradio web interface for converting PDF documents to Markdown or extracting text using the DeepSeek-OCR API. This application was developed through an iterative process to address real-world document processing needs.

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

## Features Summary

### 🚀 Core Functionality
| Feature | Description |
|---------|-------------|
| PDF to Markdown | Convert PDF documents to clean, formatted Markdown |
| OCR Mode | Extract raw text from PDF documents |
| Custom Prompt | Use custom instructions for specialized extraction |
| Multi-file Processing | Process multiple PDFs sequentially in a queue |

### 📋 Queue System
| Feature | Description |
|---------|-------------|
| Drag & Drop | Drop PDF files directly - auto-added to queue |
| Batch Upload | Add multiple files at once |
| Queue Display | Visual list showing file names, page counts, and status |
| Clear All | Remove all items from queue |
| Clear Done | Remove only completed/failed items |
| Add While Processing | Drop new files even during active processing |

### 🔄 Progress & Timing
| Feature | Description |
|---------|-------------|
| Real-time Progress | Live percentage updates during processing |
| Adaptive Timing | Learns actual processing speed and adjusts estimates |
| ETA Display | Shows estimated time remaining |
| Progress Clamping | Stops at 100% and recalibrates time/page when exceeded |
| Per-file Stats | Shows elapsed time and seconds/page for each file |

### 🧹 Post-Processing & Cleanup
| Feature | Description |
|---------|-------------|
| Tag Cleanup | Removes `<\|ref\|>`, `<\|det\|>`, `<\|/ref\|>`, `<\|/det\|>` tags |
| Incomplete Tag Handling | Cleans truncated tags at end of output |
| End-of-sentence Token | Removes `<｜end▁of▁sentence｜>` markers |
| Page Split Removal | Optional removal of `<--- Page Split --->` markers |
| LaTeX Symbol Cleanup | Converts `\coloneqq` → `:=`, `\eqqcolon` → `=:` |
| Excessive Newline Cleanup | Normalizes multiple blank lines |

### 🖼️ Image Extraction
| Feature | Description |
|---------|-------------|
| Coordinate-based Extraction | Extracts images from `<\|det\|>` coordinate tags |
| Multi-page Support | Handles images across multiple PDF pages |
| ZIP Download | All extracted images bundled in a downloadable ZIP |
| Markdown Links | Replaces image tags with markdown image references |

### 💾 File Persistence
| Feature | Description |
|---------|-------------|
| Upload Storage | All uploads saved to `data/uploads/` with timestamps |
| Result Storage | All results saved to `data/results/` as `.md` files |
| Image Storage | Extracted images saved to `data/images/` |
| Metadata | JSON metadata files for each result |
| Result Caching | Detects previously processed files |

### 📖 Output Display
| Feature | Description |
|---------|-------------|
| Rendered Tab | Beautiful markdown rendering with scrollbar |
| Raw Tab | Plain text view for copying/editing |
| Scrollable Container | Max 600px height with custom scrollbar styling |
| Multi-file Results | Concatenated results with separators |

### 📥 Downloads
| Feature | Description |
|---------|-------------|
| Markdown Files | Download individual or multiple result files |
| Images ZIP | Download all extracted images in one archive |
| Previous Results | Load and view previously saved results |

### ⏹️ Control
| Feature | Description |
|---------|-------------|
| Cancel Button | Stop processing after current file completes |
| API Health Check | Verify API connection status |
| Refresh Controls | Refresh API status and file lists |

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

## Development History & Iterations

### Phase 1: Basic Gradio Conversion
- Converted original CLI scripts to Gradio web interface
- Added basic PDF upload and processing
- Implemented Markdown/OCR/Custom mode selection

### Phase 2: Queue System
- Added multi-file queue with status tracking
- Implemented drag-and-drop auto-add functionality
- Created visual queue display with file info and page counts

### Phase 3: Progress & Timing
- Added real-time progress updates using Gradio generators
- Implemented adaptive timing that learns from actual processing
- Added ETA calculations and per-file statistics
- **Fixed:** Progress clamping at 100% with time/page recalibration

### Phase 4: Post-Processing
- Implemented tag cleanup for `<|ref|>`, `<|det|>` patterns
- Added handling for incomplete/truncated tags at document end
- Added optional page split marker removal
- Implemented LaTeX symbol replacement

### Phase 5: Image Extraction
- Added coordinate-based image extraction from PDF pages
- Implemented multi-page image distribution
- Created ZIP archive generation for downloads
- Added markdown image link replacement

### Phase 6: UI/UX Polish
- Added scrollable markdown container with custom CSS
- Implemented Rendered/Raw tabs for output viewing
- Added Clear Done button for queue management
- Improved file drop persistence (drop zone stays visible)

### Phase 7: Reliability
- Added extended timeout (2 hours) for large documents
- Implemented file persistence for uploads and results
- Added result caching to skip re-processing
- Improved error handling and logging

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
