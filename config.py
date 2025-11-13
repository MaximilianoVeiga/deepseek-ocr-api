# -*- coding: utf-8 -*-
"""Configuration module for DeepSeek-OCR API."""
import os
from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv
import torch

from constants import (
    APP_NAME,
    DEFAULT_VERSION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_BASE_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PDF_DPI,
    DEFAULT_ENABLE_COMPRESSION,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_PNG_COMPRESSION,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PDF_PAGES,
    DEFAULT_OCR_PROMPT,
    DEFAULT_CORS_ORIGINS,
    MIN_PORT,
    MAX_PORT,
    MIN_FILE_SIZE_MB,
    MAX_FILE_SIZE_MB,
    MIN_PDF_PAGES,
    MAX_PDF_PAGES,
    DEVICE_CUDA,
    DEVICE_MPS,
    DEVICE_CPU,
)

# Load environment variables from .env file
load_dotenv()


DeviceType = Literal["auto", "cuda", "mps", "cpu"]
EnvironmentType = Literal["development", "production"]


def detect_device() -> str:
    """
    Detect the best available device for model inference.
    
    Returns:
        str: Device type (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return DEVICE_CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DEVICE_MPS
    else:
        return DEVICE_CPU


class Config(BaseModel):
    """Application configuration."""
    
    model_config = ConfigDict(
        frozen=True,  # Make config immutable
        protected_namespaces=()  # Allow model_name field without warning
    )
    
    # Server configuration
    host: str = Field(default=DEFAULT_HOST, description="Server host")
    port: int = Field(default=DEFAULT_PORT, description="Server port")
    
    # Model configuration
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="DeepSeek-OCR model name"
    )
    device: str = Field(
        default="auto",
        description="Device to use for model inference (auto, cuda, mps, cpu)"
    )
    
    # OCR configuration
    default_prompt: str = Field(
        default=DEFAULT_OCR_PROMPT,
        description="Default OCR prompt"
    )
    base_size: int = Field(default=DEFAULT_BASE_SIZE, description="Base image size for OCR")
    image_size: int = Field(default=DEFAULT_IMAGE_SIZE, description="Image size for OCR")
    pdf_dpi: int = Field(default=DEFAULT_PDF_DPI, description="DPI for PDF rasterization")
    
    # Image compression configuration
    enable_compression: bool = Field(
        default=DEFAULT_ENABLE_COMPRESSION,
        description="Enable image compression for performance"
    )
    max_image_dimension: int = Field(
        default=DEFAULT_MAX_IMAGE_DIMENSION,
        description="Maximum image dimension (width/height) before resizing"
    )
    jpeg_quality: int = Field(
        default=DEFAULT_JPEG_QUALITY,
        description="JPEG compression quality (1-100)"
    )
    png_compression: int = Field(
        default=DEFAULT_PNG_COMPRESSION,
        description="PNG compression level (0-9)"
    )
    
    # File limits
    max_file_size_mb: int = Field(default=DEFAULT_MAX_FILE_SIZE_MB, description="Maximum file size in MB")
    max_pdf_pages: int = Field(default=DEFAULT_MAX_PDF_PAGES, description="Maximum PDF pages to process")
    
    # CORS configuration
    cors_origins: list[str] = Field(default=DEFAULT_CORS_ORIGINS, description="CORS allowed origins")
    
    # Security configuration
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_key: str = Field(default="", description="API key for authentication")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute per IP")
    prompt_max_length: int = Field(default=2000, description="Maximum prompt length")
    
    # Application metadata
    version: str = Field(default=DEFAULT_VERSION, description="API version")
    title: str = Field(default=APP_NAME, description="API title")
    environment: str = Field(default="development", description="Environment (development/production)")
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not MIN_PORT <= v <= MAX_PORT:
            raise ValueError(f"Port must be between {MIN_PORT} and {MAX_PORT}")
        return v
    
    @field_validator("max_file_size_mb")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max file size is reasonable."""
        if v < MIN_FILE_SIZE_MB or v > MAX_FILE_SIZE_MB:
            raise ValueError(f"Max file size must be between {MIN_FILE_SIZE_MB} and {MAX_FILE_SIZE_MB} MB")
        return v
    
    @field_validator("max_pdf_pages")
    @classmethod
    def validate_max_pdf_pages(cls, v: int) -> int:
        """Validate max PDF pages is reasonable."""
        if v < MIN_PDF_PAGES or v > MAX_PDF_PAGES:
            raise ValueError(f"Max PDF pages must be between {MIN_PDF_PAGES} and {MAX_PDF_PAGES}")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and resolve device."""
        valid_devices = ["auto", DEVICE_CUDA, DEVICE_MPS, DEVICE_CPU]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of: {', '.join(valid_devices)}")
        
        # Auto-detect device if set to "auto"
        if v == "auto":
            return detect_device()
        
        # Validate requested device is available
        if v == DEVICE_CUDA and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        if v == DEVICE_MPS and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
            raise ValueError("MPS device requested but not available")
        
        return v
    
    @field_validator("api_key_enabled")
    @classmethod
    def validate_api_key_config(cls, v: bool, info) -> bool:
        """Validate API key configuration."""
        # This validator runs before api_key field is set, so we can't check it here
        # The validation will be done in model_validator
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        import warnings
        
        # Warn if API key is enabled but not set
        if self.api_key_enabled and not self.api_key:
            warnings.warn(
                "API key authentication is enabled but no API key is configured. "
                "Set API_KEY environment variable.",
                UserWarning
            )
        
        # Warn if CORS is wide open in production
        if self.environment == "production" and "*" in self.cors_origins:
            warnings.warn(
                "CORS is set to allow all origins (*) in production environment. "
                "Consider restricting to specific domains for better security.",
                UserWarning
            )
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Config: Application configuration
    """
    return Config(
        host=os.getenv("HOST", DEFAULT_HOST),
        port=int(os.getenv("PORT", str(DEFAULT_PORT))),
        model_name=os.getenv("DS_OCR_MODEL", DEFAULT_MODEL_NAME),
        device=os.getenv("DEVICE", "auto"),
        default_prompt=os.getenv("DEFAULT_PROMPT", DEFAULT_OCR_PROMPT),
        base_size=int(os.getenv("BASE_SIZE", str(DEFAULT_BASE_SIZE))),
        image_size=int(os.getenv("IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE))),
        pdf_dpi=int(os.getenv("PDF_DPI", str(DEFAULT_PDF_DPI))),
        enable_compression=os.getenv("ENABLE_COMPRESSION", str(DEFAULT_ENABLE_COMPRESSION)).lower() in ("true", "1", "yes"),
        max_image_dimension=int(os.getenv("MAX_IMAGE_DIMENSION", str(DEFAULT_MAX_IMAGE_DIMENSION))),
        jpeg_quality=int(os.getenv("JPEG_QUALITY", str(DEFAULT_JPEG_QUALITY))),
        png_compression=int(os.getenv("PNG_COMPRESSION", str(DEFAULT_PNG_COMPRESSION))),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB))),
        max_pdf_pages=int(os.getenv("MAX_PDF_PAGES", str(DEFAULT_MAX_PDF_PAGES))),
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        api_key_enabled=os.getenv("API_KEY_ENABLED", "false").lower() in ("true", "1", "yes"),
        api_key=os.getenv("API_KEY", ""),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        prompt_max_length=int(os.getenv("PROMPT_MAX_LENGTH", "2000")),
        version=os.getenv("VERSION", DEFAULT_VERSION),
        title=os.getenv("API_TITLE", APP_NAME),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


def get_config() -> Config:
    """
    Create and return a configuration instance.
    
    Note: For dependency injection, configuration is stored in app state.
    This function is used for initialization.
    
    Returns:
        Config: Application configuration
    """
    return load_config()

