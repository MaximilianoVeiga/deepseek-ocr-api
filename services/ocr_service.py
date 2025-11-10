# -*- coding: utf-8 -*-
"""OCR service with model inference logic."""
import os
import warnings
from typing import Optional
from pathlib import Path
import contextlib
import io
import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer
import torch

from config import Config, get_config
from logger import get_logger
from models import (
    OCRProcessingError,
    FileSizeLimitError,
    PDFPageLimitError,
    validate_image_file,
    validate_pdf_file,
)
from utils import temporary_file
from constants import (
    ERROR_MODEL_NOT_LOADED,
    ERROR_MODEL_LOAD_FAILED,
    ERROR_IMAGE_INFERENCE_FAILED,
    ERROR_PDF_PROCESSING_FAILED,
    LOG_MODEL_LOADING,
    LOG_MODEL_LOADED,
    LOG_MODEL_ALREADY_LOADED,
    LOG_MODEL_LOAD_FAILED,
    LOG_PROCESSING_IMAGE,
    LOG_IMAGE_PROCESSED,
    LOG_PROCESSING_PDF,
    LOG_PDF_PAGE_COUNT,
    LOG_PROCESSING_PAGE,
    LOG_PAGE_FAILED,
    LOG_PDF_PROCESSED,
    COMPONENT_STARTUP,
    COMPONENT_OCR,
    COMPONENT_OCR_SERVICE,
    DEVICE_CUDA,
    DEVICE_CPU,
)


class OCRService:
    """
    OCR service for processing images and PDFs with DeepSeek-OCR.
    
    This service handles model loading, inference, and document processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize OCR service.
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.logger = get_logger(version=self.config.version)
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """
        Load the DeepSeek-OCR model and tokenizer.
        
        Raises:
            OCRProcessingError: If model loading fails
        """
        if self._model_loaded:
            self.logger.info(
                LOG_MODEL_ALREADY_LOADED,
                component=COMPONENT_OCR_SERVICE
            )
            return
        
        try:
            self.logger.info(
                LOG_MODEL_LOADING.format(model_name=self.config.model_name),
                component=COMPONENT_STARTUP,
                model=self.config.model_name,
                device=self.config.device
            )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                
                # Load model and move to appropriate device
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
            finally:
                # Reset warning filters
                warnings.resetwarnings()
            
            # Move model to device
            if self.config.device == DEVICE_CUDA:
                self.model = self.model.cuda().to(torch.bfloat16)
            elif self.config.device == DEVICE_CPU:
                self.model = self.model.cpu()
            else:
                # MPS or other devices
                self.model = self.model.to(self.config.device)
            
            self._model_loaded = True
            self.logger.info(
                LOG_MODEL_LOADED,
                component=COMPONENT_STARTUP,
                model=self.config.model_name,
                device=self.config.device
            )
            
        except Exception as e:
            self.logger.error(
                LOG_MODEL_LOAD_FAILED,
                component=COMPONENT_STARTUP,
                exc_info=e,
                model=self.config.model_name
            )
            raise OCRProcessingError(
                ERROR_MODEL_LOAD_FAILED.format(error=str(e)),
                original_error=e
            )
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._model_loaded or self.model is None or self.tokenizer is None:
            raise OCRProcessingError(ERROR_MODEL_NOT_LOADED)
    
    def _clean_stdout_output(self, stdout_text: str) -> str:
        """
        Clean captured stdout to extract OCR text.
        
        Filters out debug messages and extracts the actual OCR content.
        
        Args:
            stdout_text: Raw captured stdout
            
        Returns:
            str: Cleaned OCR text
        """
        if not stdout_text:
            return ""
        
        lines = stdout_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip debug lines and noise
            if any([
                line.startswith('====='),
                line.startswith('BASE:'),
                line.startswith('PATCHES:'),
                line.strip() == '(0x0)',
                line.strip().startswith('(0x0)') and len(line.strip()) < 50,
                line.strip() == '0x0',
                'torch.Size' in line,
            ]):
                continue
            
            # Keep actual content
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _infer_image(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """
        Perform OCR inference on a single image.
        
        Args:
            image_path: Path to image file
            prompt: OCR prompt
            
        Returns:
            str: Extracted text
            
        Raises:
            OCRProcessingError: If inference fails
        """
        self._ensure_model_loaded()
        
        try:
            # Create a temporary directory for model output
            # The model requires a valid output_path even when save_results=False
            import tempfile
            import shutil
            temp_output_dir = tempfile.mkdtemp(prefix="dsocr-output-")
            
            try:
                # Debug: log the prompt being used
                self.logger.info(
                    f"Calling model.infer with prompt: {prompt[:100]}...",
                    component=COMPONENT_OCR_SERVICE
                )
                
                # Capture stdout during model inference
                stdout_capture = io.StringIO()
                result = None
                
                with contextlib.redirect_stdout(stdout_capture):
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=temp_output_dir,
                        base_size=self.config.base_size,
                        image_size=self.config.image_size,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False,  # Disable compression stats
                    )
                
                # Get captured stdout
                stdout_text = stdout_capture.getvalue()
                
                # Debug: log what we got back
                self.logger.info(
                    f"Model returned type: {type(result)}, value preview: {str(result)[:200] if result else 'None'}",
                    component=COMPONENT_OCR_SERVICE
                )
                self.logger.info(
                    f"Captured stdout length: {len(stdout_text)} chars",
                    component=COMPONENT_OCR_SERVICE
                )
                
                # Priority 1: Try to extract text from captured stdout
                if stdout_text:
                    cleaned_text = self._clean_stdout_output(stdout_text)
                    if cleaned_text:
                        self.logger.info(
                            f"Successfully extracted text from stdout ({len(cleaned_text)} chars)",
                            component=COMPONENT_OCR_SERVICE
                        )
                        return cleaned_text
                
                # Priority 2: Check if result has text
                if isinstance(result, str) and result:
                    return result
                elif isinstance(result, dict):
                    # Check for common keys that might contain the text
                    for key in ['text', 'output', 'result', 'prediction']:
                        if key in result and result[key]:
                            return str(result[key])
                    self.logger.warning(
                        f"Dict returned but no text found. Keys: {list(result.keys())}",
                        component=COMPONENT_OCR_SERVICE
                    )
                elif isinstance(result, list) and result:
                    # If list, concatenate all string elements
                    text = "\n".join(str(item) for item in result if item)
                    if text:
                        return text
                
                # Priority 3: Try saving results to file as fallback
                self.logger.info(
                    "No text from stdout or result, trying with save_results=True",
                    component=COMPONENT_OCR_SERVICE
                )
                
                stdout_capture2 = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture2):
                    self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=temp_output_dir,
                        base_size=self.config.base_size,
                        image_size=self.config.image_size,
                        crop_mode=True,
                        save_results=True,
                        test_compress=False,
                    )
                
                # Try stdout from second call
                stdout_text2 = stdout_capture2.getvalue()
                if stdout_text2:
                    cleaned_text2 = self._clean_stdout_output(stdout_text2)
                    if cleaned_text2:
                        self.logger.info(
                            f"Extracted text from second stdout capture ({len(cleaned_text2)} chars)",
                            component=COMPONENT_OCR_SERVICE
                        )
                        return cleaned_text2
                
                # Look for output files in temp directory
                output_files = list(Path(temp_output_dir).glob("*.txt")) + \
                             list(Path(temp_output_dir).glob("*.md"))
                
                if output_files:
                    output_file = output_files[0]
                    self.logger.info(
                        f"Reading result from file: {output_file}",
                        component=COMPONENT_OCR_SERVICE
                    )
                    with open(output_file, "r", encoding="utf-8") as f:
                        return f.read()
                
                # Nothing worked
                self.logger.warning(
                    "Unable to extract text from model inference",
                    component=COMPONENT_OCR_SERVICE
                )
                return ""
            finally:
                # Clean up temporary output directory
                try:
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to clean up temporary output directory: {temp_output_dir}",
                        component=COMPONENT_OCR_SERVICE,
                        exc_info=cleanup_error
                    )
            
        except Exception as e:
            self.logger.error(
                "Image inference failed",
                component=COMPONENT_OCR_SERVICE,
                exc_info=e,
                image_path=image_path
            )
            raise OCRProcessingError(
                ERROR_IMAGE_INFERENCE_FAILED.format(error=str(e)),
                original_error=e
            )
    
    async def process_image(
        self,
        file_content: bytes,
        filename: str,
        prompt: str
    ) -> str:
        """
        Process an image file and extract text.
        
        Args:
            file_content: Image file content
            filename: Original filename
            prompt: OCR prompt
            
        Returns:
            str: Extracted text
            
        Raises:
            FileValidationError: If file validation fails
            FileSizeLimitError: If file size exceeds limit
            OCRProcessingError: If OCR processing fails
        """
        # Validate file
        validate_image_file(filename)
        
        # Check file size
        if len(file_content) > self.config.max_file_size_bytes:
            raise FileSizeLimitError(
                len(file_content),
                self.config.max_file_size_bytes
            )
        
        self.logger.info(
            LOG_PROCESSING_IMAGE.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            size_kb=len(file_content) // 1024
        )
        
        # Get file extension
        suffix = os.path.splitext(filename)[1] or ".png"
        
        # Process image
        with temporary_file(suffix=suffix, logger=self.logger) as tmp_path:
            # Write file content
            with open(tmp_path, "wb") as f:
                f.write(file_content)
            
            # Perform OCR
            text = self._infer_image(tmp_path, prompt)
        
        self.logger.info(
            LOG_IMAGE_PROCESSED.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            text_length=len(text)
        )
        
        # Log response text preview
        text_preview = text[:500] if len(text) > 500 else text
        self.logger.info(
            f"OCR response text (preview): {text_preview}{'...' if len(text) > 500 else ''}",
            component=COMPONENT_OCR,
            filename=filename,
            full_text_length=len(text)
        )
        
        return text
    
    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        prompt: str
    ) -> str:
        """
        Process a PDF file page by page and extract text.
        
        Args:
            file_content: PDF file content
            filename: Original filename
            prompt: OCR prompt
            
        Returns:
            str: Extracted text (pages separated by double newlines)
            
        Raises:
            FileValidationError: If file validation fails
            FileSizeLimitError: If file size exceeds limit
            PDFPageLimitError: If PDF has too many pages
            OCRProcessingError: If OCR processing fails
        """
        # Validate file
        validate_pdf_file(filename)
        
        # Check file size
        if len(file_content) > self.config.max_file_size_bytes:
            raise FileSizeLimitError(
                len(file_content),
                self.config.max_file_size_bytes
            )
        
        self.logger.info(
            LOG_PROCESSING_PDF.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            size_kb=len(file_content) // 1024
        )
        
        # Process PDF
        with temporary_file(suffix=".pdf", logger=self.logger) as pdf_path:
            # Write PDF content
            with open(pdf_path, "wb") as f:
                f.write(file_content)
            
            # Open PDF and check page count
            try:
                doc = fitz.open(pdf_path)
                page_count = doc.page_count
                
                if page_count > self.config.max_pdf_pages:
                    doc.close()
                    raise PDFPageLimitError(page_count, self.config.max_pdf_pages)
                
                self.logger.info(
                    LOG_PDF_PAGE_COUNT.format(page_count=page_count),
                    component=COMPONENT_OCR,
                    filename=filename,
                    page_count=page_count
                )
                
                # Process each page
                outputs = []
                for i in range(page_count):
                    self.logger.info(
                        LOG_PROCESSING_PAGE.format(page=i + 1, total_pages=page_count),
                        component=COMPONENT_OCR,
                        filename=filename,
                        page=i + 1,
                        total_pages=page_count
                    )
                    
                    try:
                        page = doc.load_page(i)
                        pix = page.get_pixmap(dpi=self.config.pdf_dpi)
                        
                        # Save page as image and process
                        with temporary_file(suffix=".png", logger=self.logger) as img_path:
                            pix.save(img_path)
                            
                            text = self._infer_image(img_path, prompt)
                            if text.strip():
                                outputs.append(text.strip())
                    
                    except Exception as e:
                        self.logger.error(
                            LOG_PAGE_FAILED.format(page=i + 1),
                            component=COMPONENT_OCR,
                            exc_info=e,
                            filename=filename,
                            page=i + 1
                        )
                        # Continue with other pages instead of failing completely
                        continue
                
                doc.close()
                
            except PDFPageLimitError:
                raise
            except Exception as e:
                self.logger.error(
                    "PDF processing failed",
                    component=COMPONENT_OCR,
                    exc_info=e,
                    filename=filename
                )
                raise OCRProcessingError(
                    ERROR_PDF_PROCESSING_FAILED.format(error=str(e)),
                    original_error=e
                )
        
        result_text = "\n\n".join(outputs)
        
        self.logger.info(
            LOG_PDF_PROCESSED.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            pages_processed=len(outputs),
            text_length=len(result_text)
        )
        
        # Log response text preview
        text_preview = result_text[:500] if len(result_text) > 500 else result_text
        self.logger.info(
            f"OCR response text (preview): {text_preview}{'...' if len(result_text) > 500 else ''}",
            component=COMPONENT_OCR,
            filename=filename,
            full_text_length=len(result_text),
            pages_processed=len(outputs)
        )
        
        return result_text


def get_ocr_service(config: Optional[Config] = None) -> OCRService:
    """
    Create and return an OCR service instance.
    
    Note: For dependency injection, service is stored in app state.
    This function is used for initialization.
    
    Args:
        config: Configuration object (uses default if not provided)
        
    Returns:
        OCRService: OCR service instance
    """
    return OCRService(config=config)

