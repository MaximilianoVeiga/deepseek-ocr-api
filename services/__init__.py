# -*- coding: utf-8 -*-
"""Services package for DeepSeek-OCR API."""
from .ocr_service import OCRService, get_ocr_service
from .model_loader import ModelLoader
from .text_cleaner import TextCleaner
from .inference_engine import InferenceEngine
from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor
from .image_compressor import ImageCompressor

__all__ = [
    "OCRService",
    "get_ocr_service",
    "ModelLoader",
    "TextCleaner",
    "InferenceEngine",
    "ImageProcessor",
    "PDFProcessor",
    "ImageCompressor",
]
