#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Interface

A web interface for converting PDF files to Markdown using the DeepSeek OCR API.
Features:
- Async job submission with progress tracking
- Multiple processing modes (Markdown, OCR, Custom Prompt)
- Queue system - add files one by one, process sequentially  
- Real-time progress updates with async polling
- Image extraction with ZIP download
- Auto-cleans custom tags
- File and result persistence
"""

import gradio as gr
from pathlib import Path
from typing import Optional
import logging
import time

# Import from library
from Lib import config, OCRClient, PostProcessor, FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components using the location-specific config
client = OCRClient(config)
postprocessor = PostProcessor(config)
file_manager = FileManager(config)

# Global cancel flag
cancel_requested = False

# Global queue state
file_queue = []
processing_status = {"current": "", "progress": 0, "is_processing": False}


# =============================================================================
# Queue Management Functions
# =============================================================================

def add_to_queue(pdf_files, current_queue):
    """Add files to the queue"""
    global file_queue
    
    if pdf_files is None:
        return format_queue_display(file_queue), gr.update()
    
    # Handle both single and multiple files
    files_to_add = pdf_files if isinstance(pdf_files, list) else [pdf_files]
    
    added_count = 0
    for pdf_file in files_to_add:
        if pdf_file is None:
            continue
        
        # Get source path from Gradio file object
        source_path = Path(pdf_file.name if hasattr(pdf_file, 'name') else pdf_file)
        
        # Save uploaded file using FileManager
        saved_path = file_manager.save_uploaded_file(source_path, source_path.name)
        if saved_path:
            # Get page count
            page_count = OCRClient.get_pdf_page_count(saved_path)
            
            # Add to queue
            queue_item = {
                "path": saved_path,
                "filename": Path(saved_path).name,
                "pages": page_count or "?",
                "status": "⏳ Pending",
                "job_id": None
            }
            file_queue.append(queue_item)
            added_count += 1
    
    if added_count > 0:
        logger.info(f"Added {added_count} file(s) to queue")
    
    return format_queue_display(file_queue), gr.update(value=None)


def format_queue_display(queue):
    """Format queue for display"""
    if not queue:
        return "📭 Queue is empty - drop PDF files above"
    
    display = f"📋 **Queue ({len(queue)} file(s))**\n\n"
    for i, item in enumerate(queue, 1):
        status = item.get("status", "⏳ Pending")
        filename = item.get("filename", "Unknown")
        pages = item.get("pages", "?")
        job_id = item.get("job_id", "")
        
        display += f"{i}. **{filename}** ({pages} pages) - {status}\n"
        if job_id:
            display += f"   Job ID: `{job_id}`\n"
    
    return display


def clear_queue():
    """Clear the entire queue"""
    global file_queue
    file_queue = []
    return format_queue_display(file_queue)


def clear_completed():
    """Clear completed items from queue"""
    global file_queue
    file_queue = [item for item in file_queue if item.get("status") not in ["✅ Done", "❌ Failed"]]
    return format_queue_display(file_queue)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_queue(queue, processing_mode, custom_prompt_text, extract_images, remove_page_splits):
    """Process all files in the queue with async job management - ONE AT A TIME"""
    global cancel_requested, file_queue, processing_status
    
    cancel_requested = False
    
    if not file_queue:
        yield "📭 Queue is empty", "", format_queue_display(file_queue), None, None
        return
    
    # Determine prompt
    if processing_mode == "Markdown":
        prompt = "<image>Convert the content of the image to markdown."
    elif processing_mode == "OCR":
        prompt = "<image>Extract all text from the image."
    elif processing_mode == "Custom Prompt":
        prompt = custom_prompt_text
        if not prompt or not prompt.strip():
            yield "❌ Please provide a custom prompt", "", format_queue_display(file_queue), None, None
            return
    else:
        prompt = "<image>Convert the content of the image to markdown."
    
    total_files = len(file_queue)
    all_result_files = []
    all_image_zips = []

    def wait_for_server_available(max_wait: int = 120):
        """Wait for the server to become available (single-job mode)"""
        checks = max(1, int(max_wait / config.poll_interval))
        for _ in range(checks):
            available, health_msg = client.check_available()
            if available:
                return True, health_msg
            time.sleep(config.poll_interval)
        return False, "Server still busy after waiting"
    
    # Process files ONE AT A TIME sequentially
    for file_idx, queue_item in enumerate(file_queue):
        if cancel_requested:
            queue_item["status"] = "🚫 Cancelled"
            yield f"⏹️ Cancelled by user", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            break
        
        pdf_path = queue_item["path"]
        filename = queue_item["filename"]
        
        # Update status - PROCESSING THIS FILE NOW
        queue_item["status"] = "🚀 Submitting..."
        processing_status["current"] = filename
        processing_status["is_processing"] = True
        
        status_msg = f"[{file_idx+1}/{total_files}] 📤 Submitting: {filename}"
        yield status_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None

        # Wait for server availability (single-job server)
        available, wait_msg = wait_for_server_available()
        if not available:
            queue_item["status"] = "⏳ Busy"
            yield f"❌ {wait_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            break
        else:
            queue_item["status"] = "🚀 Submitting..."
            yield f"[{file_idx+1}/{total_files}] {wait_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
        
        # Create async job - ONLY FOR THIS FILE
        job_id, error = client.create_job(pdf_path, prompt)
        
        if error:
            queue_item["status"] = f"❌ Failed (submit)"
            yield f"❌ Job creation failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
            continue
        
        queue_item["job_id"] = job_id
        queue_item["status"] = "⏳ Processing 0%..."
        yield f"[{file_idx+1}/{total_files}] ⏳ Job submitted: {job_id[:20]}...", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
        
        # WAIT FOR THIS JOB TO COMPLETE before moving to next file
        max_polls = 3600  # 2 hours max
        last_progress = -1
        
        for poll_count in range(max_polls):
            if cancel_requested:
                queue_item["status"] = "🚫 Cancelled"
                break
            
            time.sleep(config.poll_interval)
            
            status_data, error = client.get_job_status(job_id)
            if error:
                queue_item["status"] = "❌ Error (status check)"
                yield f"❌ Status check failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
            
            status = status_data.get('status')
            progress = status_data.get('progress', 0)
            processed_pages = status_data.get('processed_pages', 0)
            total_pages = status_data.get('total_pages', 0)
            
            # Update progress display
            if progress != last_progress:
                queue_item["status"] = f"⏳ {progress:.0f}%"
                status_msg = f"[{file_idx+1}/{total_files}] {filename}: {progress:.0f}% ({processed_pages}/{total_pages} pages)"
                yield status_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                last_progress = progress
            
            if status == 'completed':
                # Download result from server
                result_text, error = client.download_result(job_id)
                
                if error:
                    queue_item["status"] = "❌ Download failed"
                    yield f"❌ Download failed: {error}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                    break
                
                # Clean and process result using PostProcessor
                cleaned_text, extracted_image_paths, zip_path = postprocessor.process(
                    result_text,
                    pdf_path=pdf_path,
                    extract_images=extract_images,
                    remove_page_splits=remove_page_splits,
                    create_zip=extract_images,
                    pdf_filename=filename
                )
                
                if zip_path:
                    all_image_zips.append(zip_path)
                
                # Save result using FileManager
                result_file_path = file_manager.save_result(
                    filename,
                    cleaned_text,
                    processing_mode,
                    prompt,
                    job_id=job_id
                )
                all_result_files.append(result_file_path)
                
                queue_item["status"] = "✅ Done"
                
                result_msg = f"✅ [{file_idx+1}/{total_files}] Completed: {filename}"
                if extract_images and extracted_image_paths:
                    result_msg += f" + {len(extracted_image_paths)} images"
                
                # Return result with download files
                yield result_msg, cleaned_text, format_queue_display(file_queue), all_result_files, all_image_zips or None
                break
                
            elif status == 'failed':
                error_msg = status_data.get('error', 'Unknown error')
                queue_item["status"] = "❌ Failed"
                yield f"❌ Job failed: {error_msg}", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
                
            elif status == 'cancelled':
                queue_item["status"] = "🚫 Cancelled"
                yield f"🚫 Job was cancelled", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
                break
        
        if poll_count >= max_polls - 1:
            queue_item["status"] = "⏱️ Timeout"
            yield f"⏱️ Timeout waiting for job completion", "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None
    
    processing_status["is_processing"] = False
    
    # Final summary
    completed = len([item for item in file_queue if item.get("status") == "✅ Done"])
    failed = len([item for item in file_queue if "❌" in item.get("status", "")])
    
    final_msg = f"🎉 Processing complete! {completed} successful, {failed} failed"
    yield final_msg, "", format_queue_display(file_queue), all_result_files or None, all_image_zips or None


# =============================================================================
# UI Helper Functions
# =============================================================================

def load_previous_result(result_path):
    """Load a previous result file"""
    if not result_path:
        return "", "No file selected"
    
    content = file_manager.load_result(result_path)
    if content is not None:
        return content, f"Loaded: {Path(result_path).name}"
    else:
        return "", f"Error loading file"


def load_custom_prompt():
    """Load custom prompt from YAML file"""
    prompt = FileManager.load_custom_prompt()
    if prompt:
        return prompt
    else:
        return "❌ custom_prompt.yaml not found or invalid"


def check_api_status():
    """Check API health"""
    _, message, _ = client.check_health()
    return message


def update_prompt_visibility(processing_mode):
    """Show/hide custom prompt based on mode"""
    is_custom = (processing_mode == "Custom Prompt")
    return gr.update(visible=is_custom), gr.update(visible=is_custom)


def refresh_file_lists():
    """Refresh the list of previous results"""
    return gr.update(choices=file_manager.list_results())


def request_cancel():
    """Request cancellation of current processing"""
    global cancel_requested
    cancel_requested = True
    return "🚫 Cancellation requested - will stop after current file"


# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="DeepSeek OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 DeepSeek OCR")
    
    with gr.Row():
        with gr.Column(scale=1):
            # API Status
            api_status = gr.Textbox(
                label="API Status",
                value=check_api_status(),
                interactive=False,
                lines=3
            )
            refresh_api_btn = gr.Button("🔄 Refresh API", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("### 📤 Add Files to Queue")
            
            # File upload
            pdf_input = gr.File(
                label="Drop PDFs here",
                file_types=[".pdf"],
                file_count="multiple",
                type="file"
            )
            
            # Queue management
            with gr.Row():
                clear_queue_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                clear_completed_btn = gr.Button("✅ Clear Done", size="sm")
            
            queue_display = gr.Markdown(
                value="📭 Queue is empty - drop PDF files above",
                label="Queue"
            )
            
            gr.Markdown("---")
            gr.Markdown("### ⚙️ Settings")
            
            # Processing mode
            processing_mode = gr.Radio(
                choices=["Markdown", "OCR", "Custom Prompt"],
                value="Markdown",
                label="Processing Mode"
            )
            
            # Post-processing options
            extract_images_checkbox = gr.Checkbox(
                label="🖼️ Extract Images",
                value=False,
                info="Extract images from detected regions"
            )
            remove_page_splits_checkbox = gr.Checkbox(
                label="📄 Remove Page Splits",
                value=True,
                info="Remove '<--- Page Split --->' markers"
            )
            
            # Custom prompt
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt",
                placeholder="<image>\nYour custom instructions...",
                lines=2,
                visible=False
            )
            load_prompt_btn = gr.Button("📂 Load from YAML", size="sm", visible=False)
            
            # Process and Cancel
            with gr.Row():
                process_btn = gr.Button("🚀 Process Queue", variant="primary", size="lg")
                cancel_btn = gr.Button("⏹️ Cancel", variant="stop", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### 📂 Previous Results")
            prev_results = gr.Dropdown(
                label="Load Previous Result",
                choices=file_manager.list_results(),
                interactive=True
            )
            with gr.Row():
                load_result_btn = gr.Button("📥 Load", size="sm")
                refresh_lists_btn = gr.Button("🔄 Refresh", size="sm")
        
        with gr.Column(scale=2):
            # Progress status
            progress_status = gr.Textbox(
                label="Progress",
                value="Ready - add files and click 'Process Queue'",
                interactive=False,
                lines=1
            )
            
            # Result output with tabs
            with gr.Tabs():
                with gr.TabItem("📄 Rendered"):
                    result_rendered = gr.Markdown(
                        label="Result (Rendered)",
                        value="*Results will appear here after processing...*",
                        elem_id="rendered-output"
                    )
                
                with gr.TabItem("📝 Raw Markdown"):
                    result_text = gr.Textbox(
                        label="Result (Raw Markdown)",
                        lines=30,
                        max_lines=100,
                        value="",
                        show_copy_button=True
                    )
            
            # Downloads section
            gr.Markdown("### 📥 Downloads")
            with gr.Row():
                result_files = gr.File(
                    label="Markdown Results",
                    file_count="multiple",
                    visible=True
                )
                images_zip_file = gr.File(
                    label="Extracted Images (ZIP)",
                    file_count="multiple",
                    visible=True
                )
    
    # Event handlers
    refresh_api_btn.click(
        check_api_status,
        outputs=api_status
    )
    
    pdf_input.change(
        add_to_queue,
        inputs=[pdf_input, queue_display],
        outputs=[queue_display, pdf_input]
    )
    
    clear_queue_btn.click(
        clear_queue,
        outputs=queue_display
    )
    
    clear_completed_btn.click(
        clear_completed,
        outputs=queue_display
    )
    
    processing_mode.change(
        update_prompt_visibility,
        inputs=processing_mode,
        outputs=[custom_prompt_input, load_prompt_btn]
    )
    
    load_prompt_btn.click(
        load_custom_prompt,
        outputs=custom_prompt_input
    )
    
    process_btn.click(
        process_queue,
        inputs=[
            queue_display,
            processing_mode,
            custom_prompt_input,
            extract_images_checkbox,
            remove_page_splits_checkbox
        ],
        outputs=[progress_status, result_text, queue_display, result_files, images_zip_file]
    )
    
    result_text.change(
        lambda x: x,
        inputs=result_text,
        outputs=result_rendered
    )
    
    cancel_btn.click(
        request_cancel,
        outputs=progress_status
    )
    
    load_result_btn.click(
        load_previous_result,
        inputs=prev_results,
        outputs=[result_text, progress_status]
    )
    
    refresh_lists_btn.click(
        refresh_file_lists,
        outputs=prev_results
    )


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek-OCR GUI ({config.environment})...")
    logger.info(f"API: {config.api_base_url}, GUI port: {config.gui_port}")
    demo.queue()  # Enable queue for generator functions
    demo.launch(
        server_name=config.gui_host,
        server_port=config.gui_port,
        share=False
    )
