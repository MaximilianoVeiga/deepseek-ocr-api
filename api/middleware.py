# -*- coding: utf-8 -*-
"""Middleware for request logging and correlation tracking."""
import time
import uuid
from typing import Callable, Tuple

from starlette.types import ASGIApp, Receive, Scope, Send, Message

from logger import StructuredLogger
from constants import (
    LOG_REQUEST_RECEIVED,
    LOG_REQUEST_COMPLETED,
    COMPONENT_MIDDLEWARE,
)

# Paths to skip logging (health check endpoints)
SKIP_LOGGING_PATHS = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
})


class RequestLoggingMiddleware:
    """
    Pure ASGI middleware for logging requests and adding correlation IDs.
    
    This middleware:
    - Generates a unique correlation ID for each request
    - Logs when requests are received (skips health endpoints)
    - Logs when requests are completed with duration and status
    - Adds correlation ID to response headers
    
    Uses pure ASGI pattern for better performance than BaseHTTPMiddleware.
    """
    
    def __init__(self, app: ASGIApp, logger: StructuredLogger):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            logger: Logger instance
        """
        self.app = app
        self.logger = logger
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        ASGI interface - process request/response.
        
        Args:
            scope: ASGI connection scope
            receive: Receive channel
            send: Send channel
        """
        # Only process HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract request info from scope
        method = scope.get("method", "")
        path = scope.get("path", "")
        
        # Check if this path should skip logging
        skip_logging = path in SKIP_LOGGING_PATHS
        
        # Generate correlation ID
        correlation_id = uuid.uuid4().hex
        
        # Store correlation ID in scope state for access by handlers
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["correlation_id"] = correlation_id
        
        # Get client host
        client = scope.get("client")
        client_host = client[0] if client else None
        
        # Log request received (skip for health endpoints)
        if not skip_logging:
            self.logger.info(
                LOG_REQUEST_RECEIVED.format(method=method, path=path),
                component=COMPONENT_MIDDLEWARE,
                correlation_id=correlation_id,
                method=method,
                path=path,
                client_host=client_host,
            )
        
        # Track timing and response status
        start_time = time.perf_counter()
        status_code = 500  # Default in case of error
        
        async def send_wrapper(message: Message) -> None:
            """Wrap send to capture status and inject headers."""
            nonlocal status_code
            
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                
                # Add correlation ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message = {
                    "type": message["type"],
                    "status": status_code,
                    "headers": headers,
                }
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log request completed (skip for health endpoints)
            if not skip_logging:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                self.logger.info(
                    LOG_REQUEST_COMPLETED.format(
                        method=method,
                        path=path,
                        status_code=status_code,
                        duration_ms=duration_ms
                    ),
                    component=COMPONENT_MIDDLEWARE,
                    correlation_id=correlation_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
