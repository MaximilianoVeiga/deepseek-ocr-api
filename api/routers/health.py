# -*- coding: utf-8 -*-
"""Health check router."""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from models import HealthResponse
from services import OCRService
from api.dependencies import get_ocr_service_dependency

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Simple health check endpoint. Returns ok:true if API is operational.",
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
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Kubernetes-style readiness probe. Returns 200 if model is loaded and ready to serve requests.",
    responses={
        200: {
            "description": "Service is ready to handle requests",
            "content": {
                "application/json": {
                    "example": {"ok": True}
                }
            }
        },
        503: {
            "description": "Service not ready (model not loaded)",
            "content": {
                "application/json": {
                    "example": {"ok": False}
                }
            }
        }
    },
    response_description="Readiness status"
)
async def readiness_check(
    service: OCRService = Depends(get_ocr_service_dependency)
) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes deployments.
    
    Checks if the model is loaded and service is ready to handle requests.
    Returns 200 if ready, 503 if not ready.
    
    Args:
        service: OCR service instance
        
    Returns:
        Dict with readiness status
    """
    from fastapi import Response
    from fastapi.responses import JSONResponse
    
    is_ready = service._model_loaded and service.model is not None
    
    if is_ready:
        return {"ok": True, "model_loaded": True}
    else:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "model_loaded": False, "reason": "Model not loaded"}
        )


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Kubernetes-style liveness probe. Returns 200 if service is alive and responding.",
    responses={
        200: {
            "description": "Service is alive",
            "content": {
                "application/json": {
                    "example": {"ok": True}
                }
            }
        }
    },
    response_description="Liveness status"
)
async def liveness_check() -> HealthResponse:
    """
    Liveness probe for Kubernetes deployments.
    
    Simple endpoint that returns 200 if the service is alive and responding.
    
    Returns:
        HealthResponse: Basic status indicator
    """
    return HealthResponse(ok=True)

