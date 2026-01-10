"""
DeepSeek-OCR Library

Provides API client and post-processing utilities for DeepSeek-OCR.
Can be used by GUI, CLI scripts, or Docker-internal applications.
"""

from .config import Config, get_config
from .ocr_client import OCRClient
from .postprocessor import PostProcessor

__all__ = ['Config', 'get_config', 'OCRClient', 'PostProcessor']
