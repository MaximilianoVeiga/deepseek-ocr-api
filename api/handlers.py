# -*- coding: utf-8 -*-
"""Exception handlers for FastAPI application."""
from typing import TYPE_CHECKING
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from models import (
    ErrorResponse,
    FileValidationError,
    OCRProcessingError,
    FileSizeLimitError,
    PDFPageLimitError,
)
from logger import StructuredLogger
from constants import (
    LOG_FILE_VALIDATION_ERROR,
    LOG_FILE_SIZE_LIMIT_ERROR,
    LOG_PDF_PAGE_LIMIT_ERROR,
    LOG_OCR_PROCESSING_ERROR,
    LOG_REQUEST_VALIDATION_ERROR,
    LOG_UNEXPECTED_ERROR,
    COMPONENT_API,
    HTTP_BAD_REQUEST,
    HTTP_REQUEST_ENTITY_TOO_LARGE,
    HTTP_UNPROCESSABLE_ENTITY,
    HTTP_INTERNAL_SERVER_ERROR,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


def register_exception_handlers(app: "FastAPI", logger: StructuredLogger) -> None:
    """
    Register all exception handlers for the application.
    
    Args:
        app: FastAPI application instance
        logger: Logger instance
    """
    
    @app.exception_handler(FileValidationError)
    async def file_validation_error_handler(
        request: Request,
        exc: FileValidationError
    ) -> JSONResponse:
        """Handle file validation errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.warning(
            LOG_FILE_VALIDATION_ERROR.format(message=exc.message),
            component=COMPONENT_API,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_BAD_REQUEST,
            content=ErrorResponse(
                detail=exc.message,
                error_type="FileValidationError",
                correlation_id=correlation_id
            ).model_dump()
        )
    
    @app.exception_handler(FileSizeLimitError)
    async def file_size_limit_error_handler(
        request: Request,
        exc: FileSizeLimitError
    ) -> JSONResponse:
        """Handle file size limit errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.warning(
            LOG_FILE_SIZE_LIMIT_ERROR.format(message=exc.message),
            component=COMPONENT_API,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                detail=exc.message,
                error_type="FileSizeLimitError",
                correlation_id=correlation_id
            ).model_dump()
        )
    
    @app.exception_handler(PDFPageLimitError)
    async def pdf_page_limit_error_handler(
        request: Request,
        exc: PDFPageLimitError
    ) -> JSONResponse:
        """Handle PDF page limit errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.warning(
            LOG_PDF_PAGE_LIMIT_ERROR.format(message=exc.message),
            component=COMPONENT_API,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                detail=exc.message,
                error_type="PDFPageLimitError",
                correlation_id=correlation_id
            ).model_dump()
        )
    
    @app.exception_handler(OCRProcessingError)
    async def ocr_processing_error_handler(
        request: Request,
        exc: OCRProcessingError
    ) -> JSONResponse:
        """Handle OCR processing errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.error(
            LOG_OCR_PROCESSING_ERROR.format(message=exc.message),
            component=COMPONENT_API,
            exc_info=exc.original_error,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail=exc.message,
                error_type="OCRProcessingError",
                correlation_id=correlation_id
            ).model_dump()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.warning(
            LOG_REQUEST_VALIDATION_ERROR.format(error=str(exc)),
            component=COMPONENT_API,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=str(exc),
                error_type="ValidationError",
                correlation_id=correlation_id
            ).model_dump()
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Log based on status code severity
        if exc.status_code >= 500:
            logger.error(
                f"HTTP error {exc.status_code}: {exc.detail}",
                component=COMPONENT_API,
                path=request.url.path,
                correlation_id=correlation_id
            )
        else:
            logger.info(
                f"HTTP error {exc.status_code}: {exc.detail}",
                component=COMPONENT_API,
                path=request.url.path,
                correlation_id=correlation_id
            )
            
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=str(exc.detail),
                error_type="HTTPException",
                correlation_id=correlation_id
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected errors."""
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.error(
            LOG_UNEXPECTED_ERROR.format(error=str(exc)),
            component=COMPONENT_API,
            exc_info=exc,
            path=request.url.path,
            correlation_id=correlation_id
        )
        return JSONResponse(
            status_code=HTTP_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An unexpected error occurred",
                error_type="InternalServerError",
                correlation_id=correlation_id
            ).model_dump()
        )

