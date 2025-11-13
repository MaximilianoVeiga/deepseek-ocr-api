# -*- coding: utf-8 -*-
"""OCR processing router."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request

from models import ImageOCRResponse, PDFOCRResponse, PageResult, ErrorResponse, FileValidationError, OutputFormat
from services import OCRService
from config import Config
from api.dependencies import get_ocr_service_dependency, get_config_dependency
from constants import ERROR_FILE_REQUIRED, ERROR_FILE_EMPTY, HTTP_BAD_REQUEST, OUTPUT_FORMAT_PROMPTS

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post(
    "/image",
    response_model=ImageOCRResponse,
    summary="Extract text from images",
    description="Extract text from image files with selectable output formats (markdown, text, table, figure, json, structured_data). Supported formats: PNG, JPG, JPEG, WEBP, BMP, TIFF, TIF. Tip: Use high-resolution images (300+ DPI) for best results.",
    responses={
        200: {
            "description": "Successfully extracted text from image",
            "model": ImageOCRResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "text": "# Invoice #12345\n\n**Date:** November 10, 2024\n\n| Item | Quantity | Price |\n|------|----------|-------|\n| Widget A | 2 | $50.00 |\n| Widget B | 1 | $75.00 |\n\n**Total:** $175.00",
                        "filename": "invoice.jpg",
                        "processing_time_seconds": 2.35,
                        "model_version": "unsloth/DeepSeek-OCR",
                        "correlation_id": "abc123-def456"
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
    response_description="Extracted text with metadata"
)
async def ocr_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    output_format: OutputFormat = Form(
        default=OutputFormat.MARKDOWN,
        description="Output format for extracted text. Choose from: markdown (default), text, table, figure, json, or structured_data.",
    ),
    include_grounding: bool = Form(
        default=False,
        description="Whether to include grounding annotations (reference tags and bounding boxes) in the output. Default is False for clean text.",
    ),
    service: OCRService = Depends(get_ocr_service_dependency),
    config: Config = Depends(get_config_dependency),
) -> ImageOCRResponse:
    """
    Process an image file and extract text using OCR.
    
    Args:
        request: FastAPI request object
        file: Image file to process
        output_format: Output format for the extracted text
        service: OCR service instance
        config: Application configuration
        
    Returns:
        ImageOCRResponse: Extracted text with metadata
    """
    if not file.filename:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
    
    # Validate file size before reading content to avoid loading large files into memory
    if file.size and file.size > config.max_file_size_bytes:
        from models import FileSizeLimitError
        raise FileSizeLimitError(file.size, config.max_file_size_bytes)
    
    # Read file content
    file_content = await file.read()
    
    if not file_content:
        raise FileValidationError(ERROR_FILE_EMPTY)
    
    # Get prompt based on selected output format
    ocr_prompt = OUTPUT_FORMAT_PROMPTS.get(output_format.value, OUTPUT_FORMAT_PROMPTS["markdown"])
    
    # Process image (strip_grounding is the inverse of include_grounding)
    text, processing_time = await service.process_image(
        file_content=file_content,
        filename=file.filename,
        prompt=ocr_prompt,
        strip_grounding=not include_grounding
    )
    
    # Get correlation ID from request state
    correlation_id = getattr(request.state, "correlation_id", None)
    
    return ImageOCRResponse(
        success=True,
        text=text,
        filename=file.filename,
        processing_time_seconds=round(processing_time, 2),
        model_version=config.model_name,
        correlation_id=correlation_id
    )


@router.post(
    "/pdf",
    response_model=PDFOCRResponse,
    summary="Extract text from PDF documents",
    description="Extract text from multi-page PDFs page by page with selectable output formats (markdown, text, table, figure, json, structured_data). Limits: Max 100 pages, max 50 MB. Note: Processing time scales with page count. Failed pages are included with error information.",
    responses={
        200: {
            "description": "Successfully extracted text from PDF pages",
            "model": PDFOCRResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "pages": [
                            {
                                "page_number": 1,
                                "text": "# Contract Agreement\n\n**Date:** 2024-01-15...",
                                "processing_time_seconds": 2.10,
                                "success": True
                            },
                            {
                                "page_number": 2,
                                "text": "## Terms and Conditions\n\n1. Payment terms...",
                                "processing_time_seconds": 2.05,
                                "success": True
                            }
                        ],
                        "total_pages": 2,
                        "filename": "contract.pdf",
                        "total_processing_time_seconds": 4.15,
                        "model_version": "unsloth/DeepSeek-OCR",
                        "correlation_id": "xyz789-abc123",
                        "warnings": []
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
    response_description="Extracted text from all pages with metadata"
)
async def ocr_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    output_format: OutputFormat = Form(
        default=OutputFormat.MARKDOWN,
        description="Output format for extracted text applied to each page. Choose from: markdown (default), text, table, figure, json, or structured_data.",
    ),
    include_grounding: bool = Form(
        default=False,
        description="Whether to include grounding annotations (reference tags and bounding boxes) in the output. Default is False for clean text.",
    ),
    service: OCRService = Depends(get_ocr_service_dependency),
    config: Config = Depends(get_config_dependency),
) -> PDFOCRResponse:
    """
    Process a PDF file page by page and extract text using OCR.
    
    Args:
        request: FastAPI request object
        file: PDF file to process
        output_format: Output format for the extracted text
        service: OCR service instance
        config: Application configuration
        
    Returns:
        PDFOCRResponse: Extracted text with per-page metadata
    """
    if not file.filename:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=ERROR_FILE_REQUIRED)
    
    # Validate file size before reading content to avoid loading large files into memory
    if file.size and file.size > config.max_file_size_bytes:
        from models import FileSizeLimitError
        raise FileSizeLimitError(file.size, config.max_file_size_bytes)
    
    # Read file content
    file_content = await file.read()
    
    if not file_content:
        raise FileValidationError(ERROR_FILE_EMPTY)
    
    # Get prompt based on selected output format
    ocr_prompt = OUTPUT_FORMAT_PROMPTS.get(output_format.value, OUTPUT_FORMAT_PROMPTS["markdown"])
    
    # Process PDF (strip_grounding is the inverse of include_grounding)
    page_results, warnings = await service.process_pdf(
        file_content=file_content,
        filename=file.filename,
        prompt=ocr_prompt,
        strip_grounding=not include_grounding
    )
    
    # Get correlation ID from request state
    correlation_id = getattr(request.state, "correlation_id", None)
    
    # Convert page results to PageResult models
    pages = [PageResult(**page) for page in page_results]
    
    # Calculate total processing time
    total_time = sum(page.processing_time_seconds for page in pages)
    
    return PDFOCRResponse(
        success=all(page.success for page in pages),
        pages=pages,
        total_pages=len(pages),
        filename=file.filename,
        total_processing_time_seconds=round(total_time, 2),
        model_version=config.model_name,
        correlation_id=correlation_id,
        warnings=warnings
    )

