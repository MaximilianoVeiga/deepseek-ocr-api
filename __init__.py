# -*- coding: utf-8 -*-
"""
DeepSeek-OCR API - FastAPI service for OCR processing.

This package provides a RESTful API for processing images and PDFs
using the DeepSeek-OCR model.
"""
from config import Config, load_config
from logger import StructuredLogger
from api import create_app

__version__ = "1.0.0"
__all__ = [
    "Config",
    "load_config",
    "StructuredLogger",
    "create_app",
]
