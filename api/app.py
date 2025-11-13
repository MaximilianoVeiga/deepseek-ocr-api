# -*- coding: utf-8 -*-
"""FastAPI application factory."""
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Config, get_config
from logger import get_logger
from services import OCRService, get_ocr_service
from .handlers import register_exception_handlers
from .routers import health_router, ocr_router
from .middleware import RequestLoggingMiddleware


# API metadata for Swagger documentation
DESCRIPTION = """
High-performance OCR API powered by DeepSeek's vision-language model.

## Features
* Extract text from images (PNG, JPG, WEBP, BMP, TIFF)
* Process multi-page PDFs
* Custom prompts for output control
* GPU accelerated processing
* Markdown-formatted output

## Limits
- Max file size: 50 MB
- Max PDF pages: 100
"""

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Health check and system status",
    },
    {
        "name": "ocr",
        "description": "Text extraction from images and PDFs",
    },
]


def create_app(
    config: Optional[Config] = None,
    ocr_service: Optional[OCRService] = None
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Application configuration
        ocr_service: OCR service instance
        
    Returns:
        FastAPI: Configured application
    """
    # Get configuration
    app_config = config or get_config()
    logger = get_logger(version=app_config.version)
    
    # Create app with enhanced documentation
    app = FastAPI(
        title=app_config.title,
        version=app_config.version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        contact={
            "name": "DeepSeek-OCR API",
            "url": "https://huggingface.co/unsloth/DeepSeek-OCR",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware, logger=logger)
    
    # Get OCR service
    service = ocr_service or get_ocr_service(config=app_config)
    
    # Store config and service in app state
    app.state.config = app_config
    app.state.ocr_service = service
    app.state.logger = logger
    
    # Register exception handlers
    register_exception_handlers(app, logger)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(ocr_router)
    
    return app

