# -*- coding: utf-8 -*-
"""Image file processing."""
import os
import asyncio
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer

from config import Config
from logger import StructuredLogger
from models import (
    FileSizeLimitError,
    validate_image_file,
)
from utils import temporary_file
from constants import (
    LOG_PROCESSING_IMAGE,
    LOG_IMAGE_PROCESSED,
    COMPONENT_OCR,
)
from .inference_engine import InferenceEngine
from .image_compressor import ImageCompressor


class ImageProcessor:
    """
    Handles image file processing.
    
    Manages:
    - Image validation
    - File size checks
    - Temporary file handling
    - Process image workflow
    """
    
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
        inference_engine: InferenceEngine,
        executor: ThreadPoolExecutor
    ):
        """
        Initialize image processor.
        
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
        Async wrapper around synchronous _infer_image method.
        
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
    
    async def process_image(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[str, float]:
        """
        Process an image file and extract text.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer to use
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
        import time
        
        start_time = time.perf_counter()
        
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
        
        # Optimize I/O for small files using in-memory processing
        use_memory_processing = (
            self.config.use_memory_processing and 
            self.image_compressor and
            len(file_content) <= self.config.memory_processing_max_size_bytes
        )
        
        # Process image
        with temporary_file(suffix='.jpg' if use_memory_processing else suffix, logger=self.logger) as tmp_path:
            if use_memory_processing:
                # For small files: compress in-memory first, then write once
                compressed_bytes = self.image_compressor.compress_image_bytes(
                    file_content,
                    output_format='JPEG'
                )
                with open(tmp_path, "wb") as f:
                    f.write(compressed_bytes)
            else:
                # For large files: use traditional write-then-compress approach
                with open(tmp_path, "wb") as f:
                    f.write(file_content)
                
                # Compress image if compression is enabled
                if self.image_compressor:
                    # Compress in-place, converting to JPEG for better compression
                    self.image_compressor.compress_image_file(
                        tmp_path,
                        tmp_path,
                        force_jpeg=True
                    )
            
            # Perform OCR asynchronously
            text = await self._infer_image_async(
                model, tokenizer, tmp_path, prompt, strip_grounding
            )
        
        processing_time = time.perf_counter() - start_time
        
        self.logger.info(
            LOG_IMAGE_PROCESSED.format(filename=filename),
            component=COMPONENT_OCR,
            filename=filename,
            text_length=len(text),
            processing_time_seconds=processing_time
        )
        
        # Log response text preview only in debug mode to reduce overhead
        if self.config.log_level.lower() == 'debug':
            text_preview = text[:500] if len(text) > 500 else text
            self.logger.debug(
                f"OCR response text (preview): {text_preview}{'...' if len(text) > 500 else ''}",
                component=COMPONENT_OCR,
                filename=filename,
                full_text_length=len(text)
            )
        
        return text, processing_time

