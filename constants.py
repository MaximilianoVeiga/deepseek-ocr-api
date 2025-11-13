# -*- coding: utf-8 -*-
"""Application constants and default values."""
from enum import Enum
from typing import Final

# Application metadata
APP_NAME: Final[str] = "DeepSeek-OCR API"
DEFAULT_VERSION: Final[str] = "1.0.0"

# Server defaults
DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 3000

# Model defaults
DEFAULT_MODEL_NAME: Final[str] = "unsloth/DeepSeek-OCR"
DEFAULT_BASE_SIZE: Final[int] = 1024
DEFAULT_IMAGE_SIZE: Final[int] = 640
DEFAULT_PDF_DPI: Final[int] = 220

# File processing limits
DEFAULT_MAX_FILE_SIZE_MB: Final[int] = 50
DEFAULT_MAX_PDF_PAGES: Final[int] = 100
MIN_FILE_SIZE_MB: Final[int] = 1
MAX_FILE_SIZE_MB: Final[int] = 1000
MIN_PDF_PAGES: Final[int] = 1
MAX_PDF_PAGES: Final[int] = 1000

# Port validation
MIN_PORT: Final[int] = 1
MAX_PORT: Final[int] = 65535

# OCR defaults
DEFAULT_OCR_PROMPT: Final[str] = "<image>\n<|grounding|>Convert the document to markdown."

# Output format prompts
OUTPUT_FORMAT_PROMPTS: Final[dict[str, str]] = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown format with proper headers, lists, tables, and formatting.",
    "text": "<image>\n<|grounding|>Extract all text from the document as plain text without any formatting or structure.",
    "table": "<image>\n<|grounding|>Extract and format all tables from the document. Preserve table structure and data accurately.",
    "figure": "<image>\n<|grounding|>Identify and extract all figures, charts, and images. Provide captions, descriptions, and any associated text.",
    "json": "<image>\n<|grounding|>Extract all information from the document and output it as structured JSON data with appropriate key-value pairs.",
    "structured_data": "<image>\n<|grounding|>Extract structured information from the document including fields, labels, and values in a clear key-value format.",
}

# CORS defaults
DEFAULT_CORS_ORIGINS: Final[list[str]] = ["*"]

# Temporary file settings
TEMP_FILE_PREFIX: Final[str] = "dsocr-"

# Supported file formats
class SupportedImageFormat(str, Enum):
    """Supported image formats."""
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    WEBP = ".webp"
    BMP = ".bmp"
    TIFF = ".tiff"
    TIF = ".tif"

class SupportedDocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = ".pdf"

# MIME types
MIME_TYPE_PDF: Final[str] = "application/pdf"
MIME_TYPE_PNG: Final[str] = "image/png"
MIME_TYPE_JPEG: Final[str] = "image/jpeg"

# Error messages
ERROR_FILE_REQUIRED: Final[str] = "File is required"
ERROR_FILE_EMPTY: Final[str] = "File is empty"
ERROR_FILENAME_REQUIRED: Final[str] = "Filename is required"
ERROR_PROMPT_EMPTY: Final[str] = "Prompt cannot be empty"
ERROR_MODEL_NOT_LOADED: Final[str] = "Model not loaded. Call load_model() first."
ERROR_UNSUPPORTED_IMAGE_FORMAT: Final[str] = "Unsupported image format: {ext}. Supported formats: {supported}"
ERROR_UNSUPPORTED_PDF_FORMAT: Final[str] = "Only PDF files are supported for this endpoint"
ERROR_FILE_SIZE_EXCEEDED: Final[str] = "File size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.2f} MB)"
ERROR_PDF_PAGES_EXCEEDED: Final[str] = "PDF has {page_count} pages, which exceeds maximum allowed pages ({max_pages})"
ERROR_MODEL_LOAD_FAILED: Final[str] = "Failed to load model: {error}"
ERROR_IMAGE_INFERENCE_FAILED: Final[str] = "Image inference failed: {error}"
ERROR_PDF_PROCESSING_FAILED: Final[str] = "PDF processing failed: {error}"
ERROR_UNEXPECTED: Final[str] = "An unexpected error occurred"

# Log messages
LOG_STARTING_API: Final[str] = "Starting DeepSeek-OCR API"
LOG_SERVER_STARTING: Final[str] = "Server starting on http://{host}:{port}"
LOG_SERVER_STOPPED: Final[str] = "Server stopped by user"
LOG_SERVER_START_FAILED: Final[str] = "Failed to start server"
LOG_MODEL_LOADING: Final[str] = "Loading model: {model_name}"
LOG_MODEL_LOADED: Final[str] = "Model loaded successfully"
LOG_MODEL_ALREADY_LOADED: Final[str] = "Model already loaded"
LOG_MODEL_LOAD_FAILED: Final[str] = "Failed to load model"
LOG_PROCESSING_IMAGE: Final[str] = "Processing image: {filename}"
LOG_IMAGE_PROCESSED: Final[str] = "Image processed successfully: {filename}"
LOG_PROCESSING_PDF: Final[str] = "Processing PDF: {filename}"
LOG_PDF_PAGE_COUNT: Final[str] = "PDF has {page_count} pages"
LOG_PROCESSING_PAGE: Final[str] = "Processing page {page}/{total_pages}"
LOG_PAGE_FAILED: Final[str] = "Failed to process page {page}"
LOG_PDF_PROCESSED: Final[str] = "PDF processed successfully: {filename}"
LOG_TEMP_FILE_CLEANUP_FAILED: Final[str] = "Failed to cleanup temporary file: {path}"
LOG_FILE_VALIDATION_ERROR: Final[str] = "File validation error: {message}"
LOG_FILE_SIZE_LIMIT_ERROR: Final[str] = "File size limit exceeded: {message}"
LOG_PDF_PAGE_LIMIT_ERROR: Final[str] = "PDF page limit exceeded: {message}"
LOG_OCR_PROCESSING_ERROR: Final[str] = "OCR processing error: {message}"
LOG_REQUEST_VALIDATION_ERROR: Final[str] = "Request validation error: {error}"
LOG_UNEXPECTED_ERROR: Final[str] = "Unexpected error: {error}"
LOG_REQUEST_RECEIVED: Final[str] = "Request received: {method} {path}"
LOG_REQUEST_COMPLETED: Final[str] = "Request completed: {method} {path} - {status_code} ({duration_ms}ms)"

# Component tags for logging
COMPONENT_STARTUP: Final[str] = "startup"
COMPONENT_SHUTDOWN: Final[str] = "shutdown"
COMPONENT_OCR: Final[str] = "ocr"
COMPONENT_OCR_SERVICE: Final[str] = "ocr-service"
COMPONENT_API: Final[str] = "api"
COMPONENT_MAIN: Final[str] = "main"
COMPONENT_MIDDLEWARE: Final[str] = "middleware"

# Device types
DEVICE_CUDA: Final[str] = "cuda"
DEVICE_MPS: Final[str] = "mps"
DEVICE_CPU: Final[str] = "cpu"

# HTTP status codes
HTTP_OK: Final[int] = 200
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_REQUEST_ENTITY_TOO_LARGE: Final[int] = 413
HTTP_UNPROCESSABLE_ENTITY: Final[int] = 422
HTTP_INTERNAL_SERVER_ERROR: Final[int] = 500

