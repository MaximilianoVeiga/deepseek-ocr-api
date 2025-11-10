# -*- coding: utf-8 -*-
"""Health check router."""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from models import HealthResponse
from services import OCRService
from config import Config
from api.dependencies import get_ocr_service_dependency, get_config_dependency

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Simple health check endpoint. Returns `{\"ok\": true}` if API is operational.",
    responses={
        200: {
            "description": "API is operational",
            "content": {
                "application/json": {
                    "example": {"ok": True}
                }
            }
        }
    },
    response_description="Status"
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Simple status indicator showing API is operational
    """
    return HealthResponse(ok=True)


@router.get(
    "/health/detailed",
    response_model=Dict[str, Any],
    summary="Detailed health check",
    description="Health check with system information including version, model status, device, and configuration limits.",
    responses={
        200: {
            "description": "Detailed health status with system information",
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "version": "1.0.0",
                        "environment": "production",
                        "model": {
                            "name": "deepseek-ai/DeepSeek-OCR",
                            "loaded": True,
                            "device": "cuda"
                        },
                        "configuration": {
                            "max_file_size_mb": 50,
                            "max_pdf_pages": 100,
                            "pdf_dpi": 220
                        }
                    }
                }
            }
        }
    },
    response_description="System information"
)
async def detailed_health_check(
    service: OCRService = Depends(get_ocr_service_dependency),
    config: Config = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """
    Detailed health check endpoint with model and system information.
    
    Args:
        service: OCR service instance
        config: Application configuration
        
    Returns:
        Dict with health status, version, model info, and configuration limits
    """
    return {
        "ok": True,
        "version": config.version,
        "environment": config.environment,
        "model": {
            "name": config.model_name,
            "loaded": service._model_loaded,
            "device": config.device,
        },
        "configuration": {
            "max_file_size_mb": config.max_file_size_mb,
            "max_pdf_pages": config.max_pdf_pages,
            "pdf_dpi": config.pdf_dpi,
        },
    }

