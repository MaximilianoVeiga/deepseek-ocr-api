# -*- coding: utf-8 -*-
"""Image compression utilities for performance optimization."""
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image
import io

from logger import StructuredLogger
from constants import COMPONENT_OCR_SERVICE


class ImageCompressor:
    """
    Handles image compression for performance optimization.
    
    Compresses images to reduce:
    - File size for faster I/O
    - Memory usage during inference
    - Processing time
    
    Maintains image quality sufficient for OCR accuracy.
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        max_dimension: int = 2048,
        jpeg_quality: int = 85,
        png_compression: int = 6
    ):
        """
        Initialize image compressor.
        
        Args:
            logger: Optional logger instance
            max_dimension: Maximum width or height (images larger will be resized)
            jpeg_quality: JPEG quality (1-100, higher is better quality)
            png_compression: PNG compression level (0-9, higher is more compression)
        """
        self.logger = logger
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
    
    def _calculate_new_size(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate new image dimensions while maintaining aspect ratio.
        
        Args:
            width: Original width
            height: Original height
            
        Returns:
            tuple[int, int]: New (width, height)
        """
        if width <= self.max_dimension and height <= self.max_dimension:
            return width, height
        
        # Calculate scaling factor
        if width > height:
            scale = self.max_dimension / width
        else:
            scale = self.max_dimension / height
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return new_width, new_height
    
    def compress_image_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        force_jpeg: bool = False
    ) -> str:
        """
        Compress an image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save compressed image (if None, overwrites input)
            force_jpeg: Force conversion to JPEG format for maximum compression
            
        Returns:
            str: Path to compressed image
        """
        output_path = output_path or input_path
        
        try:
            # Open image
            with Image.open(input_path) as img:
                original_size = Path(input_path).stat().st_size
                original_dimensions = img.size
                
                # Convert RGBA to RGB if forcing JPEG
                if force_jpeg and img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if necessary
                new_width, new_height = self._calculate_new_size(img.width, img.height)
                if (new_width, new_height) != (img.width, img.height):
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    if self.logger:
                        self.logger.info(
                            f"Resized image from {original_dimensions} to {new_width}x{new_height}",
                            component=COMPONENT_OCR_SERVICE
                        )
                
                # Save with compression
                if force_jpeg or Path(input_path).suffix.lower() in ('.jpg', '.jpeg'):
                    img.save(
                        output_path,
                        'JPEG',
                        quality=self.jpeg_quality,
                        optimize=True
                    )
                else:
                    img.save(
                        output_path,
                        'PNG',
                        compress_level=self.png_compression,
                        optimize=True
                    )
                
                # Log compression results
                if self.logger and Path(output_path).exists():
                    compressed_size = Path(output_path).stat().st_size
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    self.logger.info(
                        f"Compressed image: {original_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB "
                        f"({compression_ratio:.1f}% reduction)",
                        component=COMPONENT_OCR_SERVICE,
                        original_size_kb=original_size / 1024,
                        compressed_size_kb=compressed_size / 1024,
                        compression_ratio=compression_ratio
                    )
        
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Image compression failed, using original: {str(e)}",
                    component=COMPONENT_OCR_SERVICE,
                    exc_info=e
                )
            # If compression fails, copy original if paths differ
            if input_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)
        
        return output_path
    
    def compress_image_bytes(
        self,
        image_bytes: bytes,
        output_format: str = 'JPEG'
    ) -> bytes:
        """
        Compress image from bytes.
        
        Args:
            image_bytes: Input image bytes
            output_format: Output format ('JPEG' or 'PNG')
            
        Returns:
            bytes: Compressed image bytes
        """
        try:
            # Open image from bytes
            img = Image.open(io.BytesIO(image_bytes))
            
            original_size = len(image_bytes)
            
            # Convert RGBA to RGB if saving as JPEG
            if output_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if necessary
            new_width, new_height = self._calculate_new_size(img.width, img.height)
            if (new_width, new_height) != (img.width, img.height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Compress to bytes
            output = io.BytesIO()
            if output_format == 'JPEG':
                img.save(output, 'JPEG', quality=self.jpeg_quality, optimize=True)
            else:
                img.save(output, 'PNG', compress_level=self.png_compression, optimize=True)
            
            compressed_bytes = output.getvalue()
            
            if self.logger:
                compressed_size = len(compressed_bytes)
                compression_ratio = (1 - compressed_size / original_size) * 100
                self.logger.info(
                    f"Compressed image bytes: {original_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB "
                    f"({compression_ratio:.1f}% reduction)",
                    component=COMPONENT_OCR_SERVICE
                )
            
            return compressed_bytes
        
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Image bytes compression failed, using original: {str(e)}",
                    component=COMPONENT_OCR_SERVICE,
                    exc_info=e
                )
            return image_bytes
    
    def compress_pixmap(
        self,
        pixmap,
        output_path: str,
        force_jpeg: bool = True
    ) -> str:
        """
        Compress a PyMuPDF pixmap (from PDF page).
        
        Args:
            pixmap: PyMuPDF Pixmap object
            output_path: Path to save compressed image
            force_jpeg: Force JPEG format for better compression
            
        Returns:
            str: Path to compressed image
        """
        try:
            # Convert pixmap to PIL Image
            img_bytes = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB for JPEG
            if force_jpeg and img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if necessary
            new_width, new_height = self._calculate_new_size(img.width, img.height)
            if (new_width, new_height) != (img.width, img.height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                if self.logger:
                    self.logger.info(
                        f"Resized PDF page from {img.width}x{img.height} to {new_width}x{new_height}",
                        component=COMPONENT_OCR_SERVICE
                    )
            
            # Save with compression
            if force_jpeg:
                img.save(output_path, 'JPEG', quality=self.jpeg_quality, optimize=True)
            else:
                img.save(output_path, 'PNG', compress_level=self.png_compression, optimize=True)
            
            if self.logger:
                compressed_size = Path(output_path).stat().st_size
                self.logger.info(
                    f"Compressed PDF page to {compressed_size / 1024:.1f}KB",
                    component=COMPONENT_OCR_SERVICE
                )
        
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Pixmap compression failed, using default: {str(e)}",
                    component=COMPONENT_OCR_SERVICE,
                    exc_info=e
                )
            # Fall back to default pixmap save
            pixmap.save(output_path)
        
        return output_path

