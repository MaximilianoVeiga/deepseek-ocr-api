# -*- coding: utf-8 -*-
"""API route definitions."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from models import (
    OCRResponse,
    HealthResponse,
    ErrorResponse,
    FileValidationError,
)
from services import OCRService
from config import Config
from constants import ERROR_FILE_REQUIRED, ERROR_FILE_EMPTY, HTTP_BAD_REQUEST


def register_routes(app: FastAPI, service: OCRService, config: Config) -> None:
    """
    Register all routes for the application.
    
    Args:
        app: FastAPI application instance
        service: OCR service instance
        config: Application configuration
    """
    
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Check if the API is running"
    )
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(ok=True)
    
    @app.post(
        "/ocr/image",
        response_model=OCRResponse,
        summary="OCR for images",
        description="Extract text from image files (PNG, JPG, WEBP, etc.)",
        responses={
            200: {"description": "Successful response", "model": OCRResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            413: {"description": "File too large", "model": ErrorResponse},
            500: {"description": "Processing error", "model": ErrorResponse},
        }
    )
    async def ocr_image(
        file: UploadFile = File(..., description="Image file to process"),
        prompt: str = Form(
            default=config.default_prompt,
            description="OCR prompt for the model"
        )
    ) -> OCRResponse:
        """
        Process an image file and extract text using OCR.
        
        Args:
            file: Image file to process
            prompt: OCR prompt
            
        Returns:
            OCRResponse: Extracted text
        """
        if not file.filename:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise FileValidationError(ERROR_FILE_EMPTY)
        
        # Process image
        text = await service.process_image(
            file_content=file_content,
            filename=file.filename,
            prompt=prompt
        )
        
        return OCRResponse(text=text)
    
    @app.post(
        "/ocr/pdf",
        response_model=OCRResponse,
        summary="OCR for PDFs",
        description="Extract text from PDF files page by page",
        responses={
            200: {"description": "Successful response", "model": OCRResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            413: {"description": "File too large or too many pages", "model": ErrorResponse},
            500: {"description": "Processing error", "model": ErrorResponse},
        }
    )
    async def ocr_pdf(
        file: UploadFile = File(..., description="PDF file to process"),
        prompt: str = Form(
            default=config.default_prompt,
            description="OCR prompt for the model"
        )
    ) -> OCRResponse:
        """
        Process a PDF file page by page and extract text using OCR.
        
        Args:
            file: PDF file to process
            prompt: OCR prompt
            
        Returns:
            OCRResponse: Extracted text (pages separated by double newlines)
        """
        if not file.filename:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise FileValidationError(ERROR_FILE_EMPTY)
        
        # Process PDF
        text = await service.process_pdf(
            file_content=file_content,
            filename=file.filename,
            prompt=prompt
        )
        
        return OCRResponse(text=text)

