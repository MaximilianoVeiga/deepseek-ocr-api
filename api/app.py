# -*- coding: utf-8 -*-
"""FastAPI application factory."""
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from starlette.middleware.gzip import GZipMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import Config, get_config
from logger import get_logger
from services import OCRService, get_ocr_service
from api.security import limiter
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    
    Handles startup and shutdown events for services.
    """
    # Retrieve service from app state (injected in create_app)
    service: OCRService = app.state.ocr_service
    
    # Start background worker
    await service.start()
    
    yield
    
    # Stop background worker
    await service.stop()


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
    
    # Get OCR service
    service = ocr_service or get_ocr_service(config=app_config)
    
    # Create app with enhanced documentation and lifespan
    # Use ORJSONResponse for 3-10x faster JSON serialization
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
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )
    
    # Store config and service in app state
    app.state.config = app_config
    app.state.ocr_service = service
    app.state.logger = logger
    
    # Store limiter in app state
    app.state.limiter = limiter
    
    # Add exception handler for rate limiting
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression for responses > 500 bytes
    app.add_middleware(GZipMiddleware, minimum_size=500)
    
    # Add SlowAPI middleware for rate limiting
    app.add_middleware(SlowAPIMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware, logger=logger)
    
    # Register exception handlers
    register_exception_handlers(app, logger)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(ocr_router)
    
    return app
