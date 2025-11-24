# -*- coding: utf-8 -*-
"""Middleware for request logging and correlation tracking."""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from logger import StructuredLogger
from constants import (
    LOG_REQUEST_RECEIVED,
    LOG_REQUEST_COMPLETED,
    COMPONENT_MIDDLEWARE,
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and adding correlation IDs.
    
    This middleware:
    - Generates a unique correlation ID for each request
    - Logs when requests are received
    - Logs when requests are completed with duration and status
    - Adds correlation ID to response headers
    """
    
    def __init__(self, app: ASGIApp, logger: StructuredLogger):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            logger: Logger instance
        """
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and add logging.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response: HTTP response
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        # Log request received
        self.logger.info(
            LOG_REQUEST_RECEIVED.format(
                method=request.method,
                path=request.url.path
            ),
            component=COMPONENT_MIDDLEWARE,
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )
        
        # Process request and measure duration
        start_time = time.time()
        response = await call_next(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Log request completed
        self.logger.info(
            LOG_REQUEST_COMPLETED.format(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms
            ),
            component=COMPONENT_MIDDLEWARE,
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        
        return response

