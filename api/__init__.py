# -*- coding: utf-8 -*-
"""API package for DeepSeek-OCR."""
from .app import create_app
from .dependencies import (
    get_config_dependency,
    get_logger_dependency,
    get_ocr_service_dependency,
)

__all__ = [
    "create_app",
    "get_config_dependency",
    "get_logger_dependency",
    "get_ocr_service_dependency",
]

