# -*- coding: utf-8 -*-
"""FastAPI dependencies for dependency injection."""
from fastapi import Request

from config import Config
from logger import StructuredLogger
from services import OCRService


def get_config_dependency(request: Request) -> Config:
    """
    Get configuration from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Config: Application configuration
    """
    return request.app.state.config


def get_logger_dependency(request: Request) -> StructuredLogger:
    """
    Get logger from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        StructuredLogger: Logger instance
    """
    return request.app.state.logger


def get_ocr_service_dependency(request: Request) -> OCRService:
    """
    Get OCR service from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        OCRService: OCR service instance
    """
    return request.app.state.ocr_service

