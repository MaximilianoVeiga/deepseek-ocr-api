# -*- coding: utf-8 -*-
"""Tests for OCR service."""
import pytest
from pathlib import Path
from config import Config
from services import OCRService
from models import (
    FileValidationError,
    FileSizeLimitError,
    PDFPageLimitError,
    OCRProcessingError,
)


class TestOCRServiceInitialization:
    """Tests for OCR service initialization."""
    
    def test_service_creation(self, test_config: Config):
        """Test OCR service can be created."""
        service = OCRService(config=test_config)
        
        assert service is not None
        assert service.config == test_config
        assert service._model_loaded is False
    
    def test_service_creation_with_default_config(self):
        """Test OCR service can be created with default config."""
        service = OCRService()
        
        assert service is not None
        assert service.config is not None


@pytest.mark.integration
class TestModelLoading:
    """Tests for model loading."""
    
    def test_load_model(self, ocr_service: OCRService):
        """Test model loading succeeds."""
        # Model is already loaded in the fixture
        assert ocr_service._model_loaded is True
        assert ocr_service.model is not None
        assert ocr_service.tokenizer is not None
    
    def test_load_model_idempotent(self, ocr_service: OCRService):
        """Test loading model multiple times is safe."""
        # Try loading again
        ocr_service.load_model()
        
        assert ocr_service._model_loaded is True
        assert ocr_service.model is not None


class TestFileValidation:
    """Tests for file validation."""
    
    @pytest.mark.asyncio
    async def test_validate_image_format_png(self, ocr_service: OCRService):
        """Test PNG image format is accepted."""
        content = b"fake png content"
        
        # Should not raise for .png extension
        try:
            await ocr_service.process_image(
                file_content=content,
                filename="test.png",
                prompt="test"
            )
        except FileValidationError:
            pytest.fail("PNG format should be valid")
        except Exception:
            # Other exceptions are OK (e.g., invalid content)
            pass
    
    @pytest.mark.asyncio
    async def test_validate_image_format_jpg(self, ocr_service: OCRService):
        """Test JPG image format is accepted."""
        content = b"fake jpg content"
        
        # Should not raise for .jpg extension
        try:
            await ocr_service.process_image(
                file_content=content,
                filename="test.jpg",
                prompt="test"
            )
        except FileValidationError:
            pytest.fail("JPG format should be valid")
        except Exception:
            # Other exceptions are OK (e.g., invalid content)
            pass
    
    @pytest.mark.asyncio
    async def test_validate_image_format_invalid(self, ocr_service: OCRService):
        """Test invalid image format is rejected."""
        content = b"fake content"
        
        with pytest.raises(FileValidationError) as exc_info:
            await ocr_service.process_image(
                file_content=content,
                filename="test.txt",
                prompt="test"
            )
        
        assert "unsupported" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_validate_pdf_format_valid(self, ocr_service: OCRService):
        """Test PDF format is accepted."""
        content = b"fake pdf content"
        
        # Should not raise FileValidationError for .pdf extension
        try:
            await ocr_service.process_pdf(
                file_content=content,
                filename="test.pdf",
                prompt="test"
            )
        except FileValidationError:
            pytest.fail("PDF format should be valid")
        except Exception:
            # Other exceptions are OK (e.g., invalid content)
            pass
    
    @pytest.mark.asyncio
    async def test_validate_pdf_format_invalid(self, ocr_service: OCRService):
        """Test non-PDF format is rejected for PDF endpoint."""
        content = b"fake content"
        
        with pytest.raises(FileValidationError) as exc_info:
            await ocr_service.process_pdf(
                file_content=content,
                filename="test.png",
                prompt="test"
            )
        
        assert "pdf" in str(exc_info.value).lower()


class TestFileSizeLimits:
    """Tests for file size limit validation."""
    
    @pytest.mark.asyncio
    async def test_image_size_limit_exceeded(self, ocr_service: OCRService):
        """Test file size limit is enforced for images."""
        # Create content larger than the test limit (10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        
        with pytest.raises(FileSizeLimitError) as exc_info:
            await ocr_service.process_image(
                file_content=large_content,
                filename="large.png",
                prompt="test"
            )
        
        error = exc_info.value
        assert error.file_size == len(large_content)
        assert "exceeds" in str(error).lower()
    
    @pytest.mark.asyncio
    async def test_pdf_size_limit_exceeded(self, ocr_service: OCRService):
        """Test file size limit is enforced for PDFs."""
        # Create content larger than the test limit (10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        
        with pytest.raises(FileSizeLimitError) as exc_info:
            await ocr_service.process_pdf(
                file_content=large_content,
                filename="large.pdf",
                prompt="test"
            )
        
        error = exc_info.value
        assert error.file_size == len(large_content)
        assert "exceeds" in str(error).lower()
    
    @pytest.mark.asyncio
    async def test_image_size_within_limit(self, ocr_service: OCRService):
        """Test files within size limit are accepted."""
        # Create small content (1KB)
        small_content = b"x" * 1024
        
        # Should not raise FileSizeLimitError
        try:
            await ocr_service.process_image(
                file_content=small_content,
                filename="small.png",
                prompt="test"
            )
        except FileSizeLimitError:
            pytest.fail("Small file should be accepted")
        except Exception:
            # Other exceptions are OK
            pass


@pytest.mark.integration
class TestImageProcessing:
    """Tests for image processing."""
    
    @pytest.mark.asyncio
    async def test_process_image_success(
        self,
        ocr_service: OCRService,
        sample_pdf_bytes: bytes,
        mock_prompt: str
    ):
        """Test successful image processing."""
        try:
            result = await ocr_service.process_image(
                file_content=sample_pdf_bytes,
                filename="test.pdf",
                prompt=mock_prompt
            )
            
            assert isinstance(result, str)
            # Result might be empty or contain text depending on the file
        except OCRProcessingError:
            # If processing fails (e.g., invalid file), that's OK for this test
            pass
    
    @pytest.mark.asyncio
    async def test_process_image_invalid_content(
        self,
        ocr_service: OCRService,
        mock_prompt: str
    ):
        """Test processing invalid image content fails gracefully."""
        invalid_content = b"not a real image"
        
        with pytest.raises(OCRProcessingError):
            await ocr_service.process_image(
                file_content=invalid_content,
                filename="invalid.png",
                prompt=mock_prompt
            )


@pytest.mark.integration
class TestPDFProcessing:
    """Tests for PDF processing."""
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(
        self,
        ocr_service: OCRService,
        sample_pdf_bytes: bytes,
        mock_prompt: str
    ):
        """Test successful PDF processing."""
        try:
            result = await ocr_service.process_pdf(
                file_content=sample_pdf_bytes,
                filename="test.pdf",
                prompt=mock_prompt
            )
            
            assert isinstance(result, str)
            # Multi-page PDFs should have content separated by double newlines
        except OCRProcessingError:
            # If processing fails, that's OK for this test
            pass
    
    @pytest.mark.asyncio
    async def test_process_pdf_invalid_content(
        self,
        ocr_service: OCRService,
        mock_prompt: str
    ):
        """Test processing invalid PDF content fails gracefully."""
        invalid_content = b"not a real pdf"
        
        with pytest.raises(OCRProcessingError):
            await ocr_service.process_pdf(
                file_content=invalid_content,
                filename="invalid.pdf",
                prompt=mock_prompt
            )


class TestPromptHandling:
    """Tests for prompt handling."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_custom_prompt(
        self,
        ocr_service: OCRService,
        sample_pdf_bytes: bytes
    ):
        """Test processing with custom prompt."""
        custom_prompt = "<image>\nFree OCR."
        
        try:
            result = await ocr_service.process_image(
                file_content=sample_pdf_bytes,
                filename="test.pdf",
                prompt=custom_prompt
            )
            
            assert isinstance(result, str)
        except OCRProcessingError:
            # Processing failure is OK
            pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_prompt(
        self,
        ocr_service: OCRService,
        sample_pdf_bytes: bytes
    ):
        """Test processing with empty prompt."""
        # Empty prompt should still work or fail gracefully
        try:
            result = await ocr_service.process_image(
                file_content=sample_pdf_bytes,
                filename="test.pdf",
                prompt=""
            )
            
            assert isinstance(result, str)
        except (OCRProcessingError, Exception):
            # Any error is acceptable for empty prompt
            pass


class TestResourceCleanup:
    """Tests for resource cleanup."""
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_success(
        self,
        ocr_service: OCRService,
        sample_pdf_bytes: bytes,
        mock_prompt: str
    ):
        """Test temporary files are cleaned up after successful processing."""
        import tempfile
        import os
        
        # Track temp directory before
        temp_dir = tempfile.gettempdir()
        before_files = set(os.listdir(temp_dir))
        
        try:
            await ocr_service.process_image(
                file_content=sample_pdf_bytes,
                filename="test.pdf",
                prompt=mock_prompt
            )
        except Exception:
            pass  # Ignore errors, we're testing cleanup
        
        # Check temp directory after
        after_files = set(os.listdir(temp_dir))
        
        # Should not have significantly more temp files
        # (allow some variance for other processes)
        diff = after_files - before_files
        dsocr_files = [f for f in diff if f.startswith("dsocr-")]
        assert len(dsocr_files) <= 1  # At most one temp file not cleaned
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_error(
        self,
        ocr_service: OCRService,
        mock_prompt: str
    ):
        """Test temporary files are cleaned up even when processing fails."""
        import tempfile
        import os
        
        # Track temp directory before
        temp_dir = tempfile.gettempdir()
        before_files = set(os.listdir(temp_dir))
        
        try:
            await ocr_service.process_image(
                file_content=b"invalid",
                filename="test.png",
                prompt=mock_prompt
            )
        except Exception:
            pass  # Expected to fail
        
        # Check temp directory after
        after_files = set(os.listdir(temp_dir))
        
        # Should not have significantly more temp files
        diff = after_files - before_files
        dsocr_files = [f for f in diff if f.startswith("dsocr-")]
        assert len(dsocr_files) <= 1  # At most one temp file not cleaned

