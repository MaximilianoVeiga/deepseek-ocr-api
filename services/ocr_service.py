# -*- coding: utf-8 -*-
"""OCR service with model inference logic."""
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor

from config import Config, get_config
from logger import get_logger
from models import OCRProcessingError
from constants import ERROR_MODEL_NOT_LOADED, COMPONENT_OCR_SERVICE

from .model_loader import ModelLoader
from .text_cleaner import TextCleaner
from .inference_engine import InferenceEngine
from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor


class OCRService:
    """
    OCR service for processing images and PDFs with DeepSeek-OCR.
    
    This service handles model loading, inference, and document processing.
    Uses thread pool for async execution of synchronous model inference.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize OCR service.
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.logger = get_logger(version=self.config.version)
        
        # Thread pool for running synchronous model inference
        # Use 1 worker to avoid GPU concurrency issues
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-inference")
        
        # Initialize components
        self.model_loader = ModelLoader(self.config, self.logger)
        self.text_cleaner = TextCleaner(self.logger)
        self.inference_engine = InferenceEngine(
            self.config,
            self.logger,
            self.text_cleaner
        )
        self.image_processor = ImageProcessor(
            self.config,
            self.logger,
            self.inference_engine,
            self._executor
        )
        self.pdf_processor = PDFProcessor(
            self.config,
            self.logger,
            self.inference_engine,
            self._executor
        )
    
    def load_model(self) -> None:
        """
        Load the DeepSeek-OCR model and tokenizer.
        
        Raises:
            OCRProcessingError: If model loading fails
        """
        self.model_loader.load_model()
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self.model_loader.is_loaded():
            raise OCRProcessingError(ERROR_MODEL_NOT_LOADED)
    
    async def process_image(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[str, float]:
        """
        Process an image file and extract text.
        
        Args:
            file_content: Image file content
            filename: Original filename
            prompt: OCR prompt
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            tuple[str, float]: (Extracted text, processing time in seconds)
            
        Raises:
            FileValidationError: If file validation fails
            FileSizeLimitError: If file size exceeds limit
            OCRProcessingError: If OCR processing fails
        """
        self._ensure_model_loaded()
        
        return await self.image_processor.process_image(
            self.model_loader.get_model(),
            self.model_loader.get_tokenizer(),
            file_content,
            filename,
            prompt,
            strip_grounding
        )
    
    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """
        Process a PDF file page by page and extract text.
        
        Args:
            file_content: PDF file content
            filename: Original filename
            prompt: OCR prompt
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            tuple[list[dict], list[str]]: (List of page results with timing, List of warnings)
            Each page result contains: page_number, text, processing_time_seconds, success, error (if any)
            
        Raises:
            FileValidationError: If file validation fails
            FileSizeLimitError: If file size exceeds limit
            PDFPageLimitError: If PDF has too many pages
            OCRProcessingError: If OCR processing fails
        """
        self._ensure_model_loaded()
        
        return await self.pdf_processor.process_pdf(
            self.model_loader.get_model(),
            self.model_loader.get_tokenizer(),
            file_content,
            filename,
            prompt,
            strip_grounding
        )
    
    def shutdown(self) -> None:
        """
        Shutdown the OCR service and clean up resources.
        
        This should be called when the service is no longer needed
        to ensure proper cleanup of thread pool resources.
        """
        if self._executor:
            self.logger.info(
                "Shutting down OCR service thread pool",
                component=COMPONENT_OCR_SERVICE
            )
            self._executor.shutdown(wait=True)


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
