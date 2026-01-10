"""
OCR API Client for DeepSeek-OCR.

Handles all communication with the DeepSeek-OCR API server.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any

import requests
import fitz  # PyMuPDF

from .config import Config, get_config

logger = logging.getLogger(__name__)


class OCRClient:
    """
    Client for interacting with the DeepSeek-OCR API.
    
    Supports single-job mode where the server processes one job at a time.
    The client is responsible for queuing and sequential processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the OCR client.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """Set up API endpoint URLs."""
        base = self.config.api_base_url
        self.health_endpoint = f"{base}/health"
        self.create_job_endpoint = f"{base}/jobs/create"
        self.job_status_endpoint = f"{base}/jobs/{{job_id}}"
        self.download_endpoint = f"{base}/jobs/{{job_id}}/download"
        self.metadata_endpoint = f"{base}/jobs/{{job_id}}/metadata"
    
    def check_health(self) -> Tuple[bool, str, bool]:
        """
        Check if the API is healthy.
        
        Returns:
            Tuple of (is_healthy, message, is_available)
        """
        try:
            response = requests.get(
                self.health_endpoint, 
                timeout=self.config.api_timeout_status
            )
            if response.status_code == 200:
                data = response.json()
                return True, self._format_health_info(data), data.get("available", True)
            else:
                return False, f"? API returned status code: {response.status_code}", False
        except requests.exceptions.RequestException as e:
            return False, f"? Cannot connect to API: {str(e)}", False
    
    def _format_health_info(self, data: Dict[str, Any]) -> str:
        """Format health check data for display."""
        available = data.get("available", True)
        status_icon = "?" if available else "?"
        info = f"{status_icon} API healthy | Available: {available}\n"
        info += f"Model: deepseek-ai/DeepSeek-OCR"
        current_job = data.get("current_job")
        if current_job:
            info += f"\nActive job: {current_job.get('job_id', '')} ({current_job.get('progress', 0):.0f}%)"
        return info
    
    def check_available(self) -> Tuple[bool, str]:
        """
        Check if the server is available to accept a new job.
        
        Returns:
            Tuple of (is_available, message)
        """
        ok, msg, available = self.check_health()
        return available if ok else False, msg
    
    def create_job(self, pdf_path: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Create a new OCR job.
        
        Args:
            pdf_path: Path to the PDF file
            prompt: Processing prompt
            
        Returns:
            Tuple of (job_id, error_message). job_id is None on error.
        """
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                data = {'prompt': prompt}
                
                logger.info(f"Creating job for {pdf_path}...")
                response = requests.post(
                    self.create_job_endpoint,
                    files=files,
                    data=data,
                    timeout=self.config.api_timeout_create
                )
            
            if response.status_code == 503:
                return None, "Server busy - already processing a job"
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}\n{response.text}"
            
            result = response.json()
            if result.get('success'):
                job_id = result.get('job_id')
                logger.info(f"Job created: {job_id}")
                return job_id, None
            else:
                return None, "Job creation failed"
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout creating job for {pdf_path}")
            return None, "Timeout: Server took too long to respond"
        except Exception as e:
            logger.error(f"Error creating job: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def get_job_status(self, job_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Tuple of (status_data, error_message). status_data is None on error.
        """
        try:
            url = self.job_status_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=self.config.api_timeout_status)
            
            if response.status_code == 404:
                return None, "Job not found"
            elif response.status_code != 200:
                return None, f"Error: Status code {response.status_code}"
            
            return response.json(), None
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout getting status for job {job_id}")
            return None, "Timeout: Server took too long to respond"
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def download_result(self, job_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Download the result of a completed job.
        
        Args:
            job_id: The job ID to download
            
        Returns:
            Tuple of (result_text, error_message). result_text is None on error.
        """
        try:
            url = self.download_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=self.config.api_timeout_download)
            
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}\n{response.text}"
            
            logger.info(f"Result downloaded for job {job_id}")
            return response.content.decode('utf-8'), None
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading result for job {job_id}")
            return None, "Timeout: Server took too long to respond"
        except Exception as e:
            logger.error(f"Error downloading result: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def get_job_metadata(self, job_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get metadata for a completed job.
        
        Args:
            job_id: The job ID
            
        Returns:
            Tuple of (metadata, error_message). metadata is None on error.
        """
        try:
            url = self.metadata_endpoint.format(job_id=job_id)
            response = requests.get(url, timeout=self.config.api_timeout_status)
            
            if response.status_code != 200:
                return None, f"Error: Status code {response.status_code}"
            
            return response.json(), None
        
        except Exception as e:
            logger.error(f"Error getting job metadata: {str(e)}")
            return None, f"Error: {str(e)}"
    
    @staticmethod
    def get_pdf_page_count(pdf_path: str) -> Optional[int]:
        """
        Get the number of pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages, or None on error
        """
        try:
            doc = fitz.open(pdf_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Could not get page count for {pdf_path}: {e}")
            return None
