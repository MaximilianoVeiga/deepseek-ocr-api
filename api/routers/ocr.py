# -*- coding: utf-8 -*-
"""OCR processing router."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from models import OCRResponse, ErrorResponse, FileValidationError
from services import OCRService
from config import Config
from api.dependencies import get_ocr_service_dependency, get_config_dependency
from constants import ERROR_FILE_REQUIRED, ERROR_FILE_EMPTY, HTTP_BAD_REQUEST

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post(
    "/image",
    response_model=OCRResponse,
    summary="Extract text from images",
    description="""
    Extract text from image files in markdown format.
    
    **Supported:** PNG, JPG, JPEG, WEBP, BMP, TIFF, TIF
    
    **Tips:** Use high-resolution images (300+ DPI), provide custom prompts for specific output formats.
    """,
    responses={
        200: {
            "description": "Successfully extracted text from image",
            "model": OCRResponse,
            "content": {
                "application/json": {
                    "example": {
                        "text": "# Invoice #12345\n\n**Date:** November 10, 2024\n\n| Item | Quantity | Price |\n|------|----------|-------|\n| Widget A | 2 | $50.00 |\n| Widget B | 1 | $75.00 |\n\n**Total:** $175.00"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request - missing file or unsupported format",
            "model": ErrorResponse
        },
        413: {
            "description": "File size exceeds maximum limit",
            "model": ErrorResponse
        },
        500: {
            "description": "Server error during OCR processing",
            "model": ErrorResponse
        },
    },
    response_description="Extracted text"
)
async def ocr_image(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: str = Form(default=None, description="Custom OCR prompt (optional, uses default if not provided)"),
    service: OCRService = Depends(get_ocr_service_dependency),
    config: Config = Depends(get_config_dependency),
) -> OCRResponse:
    """
    Process an image file and extract text using OCR.
    
    Args:
        file: Image file to process
        prompt: OCR prompt (uses default if not provided)
        service: OCR service instance
        config: Application configuration
        
    Returns:
        OCRResponse: Extracted text in markdown format
    """
    if not file.filename:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
    
    # Read file content
    file_content = await file.read()
    
    if not file_content:
        raise FileValidationError(ERROR_FILE_EMPTY)
    
    # Use default prompt if not provided
    ocr_prompt = prompt if prompt else config.default_prompt
    
    # Process image
    text = await service.process_image(
        file_content=file_content,
        filename=file.filename,
        prompt=ocr_prompt
    )
    
    return OCRResponse(text=text)


@router.post(
    "/pdf",
    response_model=OCRResponse,
    summary="Extract text from PDF documents",
    description="""
    Extract text from multi-page PDFs page by page. Results are separated by double newlines.
    
    **Limits:** Max 100 pages, max 50 MB
    
    **Note:** Processing time scales with page count. Failed pages are skipped.
    """,
    responses={
        200: {
            "description": "Successfully extracted text from all PDF pages",
            "model": OCRResponse,
            "content": {
                "application/json": {
                    "example": {
                        "text": "# Page 1\n\n## Chapter Introduction\n\nThis is the content from the first page...\n\n# Page 2\n\n## Chapter 2\n\nThis is the content from the second page..."
                    }
                }
            }
        },
        400: {
            "description": "Invalid request - missing file or unsupported format",
            "model": ErrorResponse
        },
        413: {
            "description": "File size exceeds limit or PDF has too many pages",
            "model": ErrorResponse
        },
        500: {
            "description": "Server error during PDF processing",
            "model": ErrorResponse
        },
    },
    response_description="Extracted text from all pages"
)
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF file to process"),
    prompt: str = Form(default=None, description="Custom OCR prompt applied to each page (optional)"),
    service: OCRService = Depends(get_ocr_service_dependency),
    config: Config = Depends(get_config_dependency),
) -> OCRResponse:
    """
    Process a PDF file page by page and extract text using OCR.
    
    Args:
        file: PDF file to process
        prompt: OCR prompt (uses default if not provided)
        service: OCR service instance
        config: Application configuration
        
    Returns:
        OCRResponse: Extracted text with pages separated by double newlines
    """
    if not file.filename:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
    
    # Read file content
    file_content = await file.read()
    
    if not file_content:
        raise FileValidationError(ERROR_FILE_EMPTY)
    
    # Use default prompt if not provided
    ocr_prompt = prompt if prompt else config.default_prompt
    
    # Process PDF
    text = await service.process_pdf(
        file_content=file_content,
        filename=file.filename,
        prompt=ocr_prompt
    )
    
    return OCRResponse(text=text)

