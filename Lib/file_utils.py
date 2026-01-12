"""File management utilities for DeepSeek-OCR."""

import hashlib
import shutil
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from .config import Config

logger = logging.getLogger(__name__)


class FileManager:
    """File management for uploads, results, and file listing."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], length: int = 12) -> str:
        """Get MD5 hash of file (truncated to length chars)."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:length]
    
    def save_uploaded_file(
        self, 
        source_path: Union[str, Path],
        original_filename: Optional[str] = None
    ) -> Optional[str]:
        """Save uploaded file with unique timestamp+hash name. Returns path or None on error."""
        try:
            source_path = Path(source_path)
            filename = original_filename or source_path.name
            file_hash = self.get_file_hash(source_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{file_hash}_{filename}"
            dest_path = self.config.upload_dir / new_filename
            shutil.copy(source_path, dest_path)
            logger.info(f"Saved uploaded file: {dest_path}")
            return str(dest_path)
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return None
    
    def save_result(
        self,
        filename: str,
        result_text: str,
        mode: str,
        prompt: str,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save processing result to the results directory.
        
        Creates both a markdown file and a JSON metadata file.
        
        Args:
            filename: Original filename
            result_text: The processed text content
            mode: Processing mode used
            prompt: Prompt used for processing
            job_id: Optional job ID from the server
            metadata: Optional additional metadata
            
        Returns:
            Path to the saved result file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        result_filename = f"{base_name}_{timestamp}_MD.md"
        result_path = self.config.results_dir / result_filename
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        # Save metadata
        meta_filename = f"{base_name}_{timestamp}_MD_meta.json"
        meta_path = self.config.results_dir / meta_filename
        
        meta_data = {
            "original_filename": filename,
            "processing_mode": mode,
            "prompt_used": prompt,
            "timestamp": timestamp,
            "result_file": result_filename,
            "job_id": job_id
        }
        if metadata:
            meta_data.update(metadata)
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        
        logger.info(f"Saved result: {result_path}")
        return str(result_path)
    
    def list_results(self, limit: int = 50) -> List[str]:
        """
        List previous result files.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result file paths, sorted by modification time (newest first)
        """
        try:
            results = list(self.config.results_dir.glob("*_MD.md"))
            results.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return [str(r) for r in results[:limit]]
        except Exception as e:
            logger.error(f"Error listing results: {str(e)}")
            return []
    
    def load_result(self, result_path: Union[str, Path]) -> Optional[str]:
        """
        Load a result file.
        
        Args:
            result_path: Path to the result file
            
        Returns:
            File contents, or None on error
        """
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading result: {str(e)}")
            return None
    
    def load_result_metadata(self, result_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a result file.
        
        Args:
            result_path: Path to the result file
            
        Returns:
            Metadata dictionary, or None if not found
        """
        try:
            result_path = Path(result_path)
            meta_path = result_path.with_name(
                result_path.stem.replace('_MD', '_MD_meta') + '.json'
            )
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    @staticmethod
    def load_custom_prompt(yaml_path: Union[str, Path] = "custom_prompt.yaml") -> Optional[str]:
        """
        Load custom prompt from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file
            
        Returns:
            Prompt string, or None if not found/error
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(f"Custom prompt file not found: {yaml_path}")
            return None
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'prompt' in data:
                    return data['prompt']
                else:
                    logger.warning(f"No 'prompt' key found in {yaml_path}")
                    return None
        except Exception as e:
            logger.error(f"Error loading custom prompt: {str(e)}")
            return None
    
    def cleanup_old_uploads(self, max_age_days: int = 7) -> int:
        """
        Remove old uploaded files.
        
        Args:
            max_age_days: Maximum age in days before deletion
            
        Returns:
            Number of files deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        try:
            for file_path in self.config.upload_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                    file_path.unlink()
                    deleted += 1
                    logger.info(f"Deleted old upload: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up uploads: {str(e)}")
        
        return deleted
