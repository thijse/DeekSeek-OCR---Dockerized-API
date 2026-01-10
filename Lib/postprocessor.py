"""
Post-processing utilities for DeepSeek-OCR output.

Handles:
- Tag cleanup (ref, det, image tags)
- Image extraction from PDF based on coordinates
- Page split removal
- Special character cleanup
"""

import re
import io
import zipfile
import urllib.parse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

from .config import Config, get_config

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processor for DeepSeek-OCR output.
    
    Cleans up OCR output by removing special tags, extracting images,
    and performing various text cleanups.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the post-processor.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
    
    @staticmethod
    def match_tags(text: str) -> Tuple[List, List[str], List[str]]:
        """
        Match reference patterns in the text.
        
        Args:
            text: The text to search
            
        Returns:
            Tuple of (all_matches, image_matches, other_matches)
        """
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        matches_image = []
        matches_other = []
        
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        
        return matches, matches_image, matches_other
    
    def pdf_to_images(self, pdf_path: str, dpi: Optional[int] = None) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for rendering (default: from config)
            
        Returns:
            List of PIL Image objects
        """
        dpi = dpi or self.config.default_dpi
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
        
        return images
    
    def extract_and_save_images(
        self, 
        pdf_path: str, 
        content: str,
        output_dir: Optional[Path] = None
    ) -> Tuple[str, List[str]]:
        """
        Extract images from content based on coordinate tags.
        
        Args:
            pdf_path: Path to the source PDF
            content: OCR output content with image tags
            output_dir: Directory to save images (default: from config)
            
        Returns:
            Tuple of (modified_content, list_of_extracted_image_paths)
        """
        output_dir = output_dir or self.config.images_dir
        extracted_paths = []
        
        pdf_images = self.pdf_to_images(pdf_path)
        if not pdf_images:
            _, matches_images, _ = self.match_tags(content)
            for tag in matches_images:
                content = content.replace(tag, '[Image]', 1)
            return content, []
        
        _, matches_images, _ = self.match_tags(content)
        total_extracted = 0
        
        for img_idx, img_tag in enumerate(matches_images):
            try:
                pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
                det_match = re.search(pattern, img_tag)
                
                if det_match:
                    det_content = det_match.group(1)
                    try:
                        coordinates = eval(det_content)
                        page_to_use = img_idx % len(pdf_images) if len(pdf_images) > 1 else 0
                        page_image = pdf_images[page_to_use]
                        image_width, image_height = page_image.size
                        
                        for points in coordinates:
                            x1, y1, x2, y2 = points
                            x1 = int(x1 / 999 * image_width)
                            y1 = int(y1 / 999 * image_height)
                            x2 = int(x2 / 999 * image_width)
                            y2 = int(y2 / 999 * image_height)
                            
                            if x1 >= x2 or y1 >= y2:
                                continue
                            
                            cropped = page_image.crop((x1, y1, x2, y2))
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            image_filename = f"{Path(pdf_path).stem}_img{total_extracted}_{timestamp}.jpg"
                            image_path = output_dir / image_filename
                            cropped.save(image_path)
                            extracted_paths.append(str(image_path))
                            
                            encoded_filename = urllib.parse.quote(image_filename)
                            markdown_link = f"\n![Extracted Image](images/{encoded_filename})\n"
                            content = content.replace(img_tag, markdown_link, 1)
                            
                            total_extracted += 1
                            break
                    except Exception as e:
                        logger.error(f"Error processing image coordinates: {str(e)}")
                        content = content.replace(img_tag, '[Image - extraction failed]', 1)
            except Exception as e:
                logger.error(f"Error extracting image: {str(e)}")
                content = content.replace(img_tag, '[Image - error]', 1)
        
        return content, extracted_paths
    
    def create_images_zip(
        self, 
        image_paths: List[str], 
        pdf_filename: str,
        output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Create a zip file containing extracted images.
        
        Args:
            image_paths: List of image file paths
            pdf_filename: Original PDF filename (used for naming the zip)
            output_dir: Directory to save the zip (default: from config)
            
        Returns:
            Path to the created zip file, or None on error
        """
        if not image_paths:
            return None
        
        output_dir = output_dir or self.config.results_dir
        base_name = Path(pdf_filename).stem
        zip_filename = f"{base_name}_images.zip"
        zip_path = output_dir / zip_filename
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for img_path in image_paths:
                    zipf.write(img_path, Path(img_path).name)
            
            logger.info(f"Created images zip: {zip_path} with {len(image_paths)} images")
            return str(zip_path)
        except Exception as e:
            logger.error(f"Error creating images zip: {str(e)}")
            return None
    
    def clean_content(
        self,
        content: str,
        extract_images: bool = False,
        pdf_path: Optional[str] = None,
        remove_page_splits: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Clean up OCR content by removing special tags.
        
        Args:
            content: Raw OCR output
            extract_images: Whether to extract images from coordinate tags
            pdf_path: Path to source PDF (required if extract_images=True)
            remove_page_splits: Whether to remove page split markers
            
        Returns:
            Tuple of (cleaned_content, list_of_extracted_image_paths)
        """
        extracted_images = []
        
        if not content:
            return content, []
        
        # Remove end-of-sentence token
        if '<?end?of?sentence?>' in content:
            content = content.replace('<?end?of?sentence?>', '')
        
        # Handle image tags
        if extract_images and pdf_path:
            content, extracted_images = self.extract_and_save_images(pdf_path, content)
        else:
            _, matches_images, _ = self.match_tags(content)
            for tag in matches_images:
                content = content.replace(tag, '', 1)
        
        # Remove other ref/det tags
        _, _, matches_other = self.match_tags(content)
        for tag in matches_other:
            content = content.replace(tag, '')
        
        # Clean up incomplete/truncated tags at end of content
        content = re.sub(r'<\|ref\|>[^<]*$', '', content)
        content = re.sub(r'<\|det\|>[^<]*$', '', content)
        content = re.sub(r'<\|ref\|>\w+<\|/ref\|><\|det\|>\[\[[\d\s,\.]*$', '', content)
        content = re.sub(r'<\|ref\|>(?![^<]*<\|/ref\|>)', '', content)
        content = re.sub(r'<\|det\|>(?![^<]*<\|/det\|>)', '', content)
        
        # Remove page splits if requested
        if remove_page_splits:
            content = re.sub(r'\n*<-+\s*Page\s*Split\s*-+>\n*', '\n\n', content, flags=re.IGNORECASE)
        
        # Clean up LaTeX symbols
        content = content.replace('\\coloneqq', ':=')
        content = content.replace('\\eqqcolon', '=:')
        
        # Normalize excessive newlines
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        content = content.replace('\n\n\n', '\n\n')
        
        return content.strip(), extracted_images
    
    def process(
        self,
        content: str,
        pdf_path: Optional[str] = None,
        extract_images: bool = False,
        remove_page_splits: bool = False,
        create_zip: bool = False,
        pdf_filename: Optional[str] = None
    ) -> Tuple[str, List[str], Optional[str]]:
        """
        Full post-processing pipeline.
        
        Args:
            content: Raw OCR output
            pdf_path: Path to source PDF
            extract_images: Whether to extract images
            remove_page_splits: Whether to remove page split markers
            create_zip: Whether to create a zip of extracted images
            pdf_filename: Original filename (for zip naming)
            
        Returns:
            Tuple of (cleaned_content, image_paths, zip_path)
        """
        cleaned_content, image_paths = self.clean_content(
            content,
            extract_images=extract_images,
            pdf_path=pdf_path,
            remove_page_splits=remove_page_splits
        )
        
        zip_path = None
        if create_zip and image_paths and pdf_filename:
            zip_path = self.create_images_zip(image_paths, pdf_filename)
        
        return cleaned_content, image_paths, zip_path
