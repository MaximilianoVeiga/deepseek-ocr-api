# -*- coding: utf-8 -*-
"""PDF document processing."""
import asyncio
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer

from config import Config
from logger import StructuredLogger
from models import (
    FileSizeLimitError,
    PDFPageLimitError,
    OCRProcessingError,
    validate_pdf_file,
)
from utils import temporary_file
from constants import (
    ERROR_PDF_PROCESSING_FAILED,
    LOG_PROCESSING_PDF,
    LOG_PDF_PAGE_COUNT,
    LOG_PROCESSING_PAGE,
    LOG_PAGE_FAILED,
    LOG_PDF_PROCESSED,
    COMPONENT_OCR,
)
from .inference_engine import InferenceEngine
from .image_compressor import ImageCompressor


class PDFProcessor:
    """
    Handles PDF document processing.
    
    Manages:
    - PDF page extraction with PyMuPDF
    - Page limit validation
    - Per-page processing with error handling
    - Results aggregation
    """
    
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
        inference_engine: InferenceEngine,
        executor: ThreadPoolExecutor
    ):
        """
        Initialize PDF processor.
        
        Args:
            config: Configuration object
            logger: Logger instance
            inference_engine: Inference engine instance
            executor: Thread pool executor for async operations
        """
        self.config = config
        self.logger = logger
        self.inference_engine = inference_engine
        self.executor = executor
        
        # Initialize image compressor if compression is enabled
        self.image_compressor = None
        if config.enable_compression:
            self.image_compressor = ImageCompressor(
                logger=logger,
                max_dimension=config.max_image_dimension,
                jpeg_quality=config.jpeg_quality,
                png_compression=config.png_compression
            )
    
    async def _infer_image_async(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        image_path: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> str:
        """
        Async wrapper around synchronous inference.
        
        Runs the synchronous model inference in a thread pool to avoid
        blocking the async event loop.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer to use
            image_path: Path to image file
            prompt: OCR prompt
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            str: Extracted text
            
        Raises:
            OCRProcessingError: If inference fails
        """
        # Run synchronous inference in thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.inference_engine.infer_image,
            model,
            tokenizer,
            image_path,
            prompt,
            strip_grounding
        )
    
    async def process_pdf(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """
        Process a PDF file page by page and extract text.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer to use
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
        import time
        
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
        page_results = []
        warnings = []
        
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
                for i in range(page_count):
                    page_start_time = time.perf_counter()
                    page_num = i + 1
                    
                    self.logger.info(
                        LOG_PROCESSING_PAGE.format(page=page_num, total_pages=page_count),
                        component=COMPONENT_OCR,
                        filename=filename,
                        page=page_num,
                        total_pages=page_count
                    )
                    
                    try:
                        page = doc.load_page(i)
                        pix = page.get_pixmap(dpi=self.config.pdf_dpi)
                        
                        # Save page as image and process
                        # Use JPEG extension for compressed output
                        suffix = ".jpg" if self.image_compressor else ".png"
                        with temporary_file(suffix=suffix, logger=self.logger) as img_path:
                            # Compress PDF page to image if compression enabled
                            if self.image_compressor:
                                self.image_compressor.compress_pixmap(
                                    pix, img_path, force_jpeg=True
                                )
                            else:
                                pix.save(img_path)
                            
                            text = await self._infer_image_async(
                                model, tokenizer, img_path, prompt, strip_grounding
                            )
                            page_time = time.perf_counter() - page_start_time
                            
                            page_results.append({
                                "page_number": page_num,
                                "text": text.strip() if text else "",
                                "processing_time_seconds": round(page_time, 2),
                                "success": True,
                                "error": None
                            })
                    
                    except Exception as e:
                        page_time = time.perf_counter() - page_start_time
                        error_msg = f"Failed to process page {page_num}: {str(e)}"
                        
                        self.logger.error(
                            LOG_PAGE_FAILED.format(page=page_num),
                            component=COMPONENT_OCR,
                            exc_info=e,
                            filename=filename,
                            page=page_num
                        )
                        
                        warnings.append(error_msg)
                        page_results.append({
                            "page_number": page_num,
                            "text": "",
                            "processing_time_seconds": round(page_time, 2),
                            "success": False,
                            "error": error_msg
                        })
                
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
        
        successful_pages = sum(1 for p in page_results if p["success"])
        total_text_length = sum(len(p["text"]) for p in page_results)
        
        self.logger.info(
            LOG_PDF_PROCESSED.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            pages_processed=successful_pages,
            total_pages=len(page_results),
            text_length=total_text_length
        )
        
        return page_results, warnings

