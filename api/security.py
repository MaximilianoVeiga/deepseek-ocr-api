# -*- coding: utf-8 -*-
"""Security utilities for API key authentication and rate limiting."""
from typing import Optional
from fastapi import Security, HTTPException, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from constants import HTTP_BAD_REQUEST, HTTP_INTERNAL_SERVER_ERROR


# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate Limiter Singleton
limiter = Limiter(key_func=get_remote_address)


def get_rate_limiter() -> Limiter:
    """
    Get the global rate limiter instance.
    
    Returns:
        Limiter: Configured rate limiter
    """
    return limiter


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    request: Request = None
) -> Optional[str]:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        request: FastAPI request object
        
    Returns:
        str: Verified API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Get config from app state
    config = request.app.state.config if request else None
    
    if not config:
        raise HTTPException(
            status_code=HTTP_INTERNAL_SERVER_ERROR,
            detail="Configuration not available"
        )
    
    # If API key authentication is not enabled, allow all requests
    if not config.api_key_enabled:
        return None
    
    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="API key required. Provide X-API-Key header."
        )
    
    # Verify API key
    if api_key != config.api_key:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="Invalid API key"
        )
    
    return api_key


def validate_prompt(prompt: str, max_length: int = 2000) -> None:
    """
    Validate and sanitize OCR prompt.
    
    Checks for suspicious patterns and enforces length limits.
    
    Args:
        prompt: OCR prompt to validate
        max_length: Maximum allowed prompt length
        
    Raises:
        HTTPException: If prompt is invalid or suspicious
    """
    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="Prompt cannot be empty"
        )
    
    if len(prompt) > max_length:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail=f"Prompt exceeds maximum length of {max_length} characters"
        )
    
    # Check for suspicious patterns (basic prompt injection detection)
    suspicious_patterns = [
        "ignore previous instructions",
        "disregard",
        "forget everything",
        "new instructions",
        "system:",
        "assistant:",
    ]
    
    prompt_lower = prompt.lower()
    for pattern in suspicious_patterns:
        if pattern in prompt_lower:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail=f"Prompt contains suspicious pattern: '{pattern}'"
            )
