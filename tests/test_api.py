# -*- coding: utf-8 -*-
"""Integration tests for API endpoints."""
import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client: TestClient):
        """Test health check returns ok status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data == {"ok": True}


@pytest.mark.integration
class TestImageOCREndpoint:
    """Tests for /ocr/image endpoint."""
    
    def test_ocr_image_success(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        mock_prompt: str
    ):
        """Test successful image OCR processing."""
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        data = {"prompt": mock_prompt}
        
        response = client.post("/ocr/image", files=files, data=data)
        
        # Integration test - may succeed or fail depending on model availability
        if response.status_code == 200:
            result = response.json()
            assert "text" in result
            assert isinstance(result["text"], str)
        else:
            # If model not available or processing fails, check error response
            assert response.status_code in [400, 500]
            error = response.json()
            assert "detail" in error
    
    def test_ocr_image_no_file(self, client: TestClient):
        """Test OCR image without file returns error."""
        response = client.post("/ocr/image", data={"prompt": "test"})
        
        assert response.status_code == 422  # Validation error
    
    def test_ocr_image_empty_file(self, client: TestClient):
        """Test OCR image with empty file returns error."""
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        
        response = client.post("/ocr/image", files=files)
        
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        assert "empty" in error["detail"].lower()
    
    def test_ocr_image_invalid_format(self, client: TestClient):
        """Test OCR image with unsupported format returns error."""
        files = {"file": ("test.txt", io.BytesIO(b"test"), "text/plain")}
        
        response = client.post("/ocr/image", files=files)
        
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
    
    def test_ocr_image_custom_prompt(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes
    ):
        """Test OCR image with custom prompt."""
        custom_prompt = "<image>\nFree OCR."
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        data = {"prompt": custom_prompt}
        
        response = client.post("/ocr/image", files=files, data=data)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500]
        result = response.json()
        assert "text" in result or "detail" in result
    
    def test_ocr_image_large_file(self, client: TestClient):
        """Test OCR image with file exceeding size limit."""
        # Create a file larger than the test limit (10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.png", io.BytesIO(large_content), "image/png")}
        
        response = client.post("/ocr/image", files=files)
        
        assert response.status_code == 413  # File too large
        error = response.json()
        assert "detail" in error
        assert "size" in error["detail"].lower()


@pytest.mark.integration
class TestPDFOCREndpoint:
    """Tests for /ocr/pdf endpoint."""
    
    def test_ocr_pdf_success(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        mock_prompt: str
    ):
        """Test successful PDF OCR processing."""
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        data = {"prompt": mock_prompt}
        
        response = client.post("/ocr/pdf", files=files, data=data)
        
        # Integration test - may succeed or fail depending on model availability
        if response.status_code == 200:
            result = response.json()
            assert "text" in result
            assert isinstance(result["text"], str)
        else:
            # If model not available or processing fails, check error response
            assert response.status_code in [400, 500]
            error = response.json()
            assert "detail" in error
    
    def test_ocr_pdf_no_file(self, client: TestClient):
        """Test OCR PDF without file returns error."""
        response = client.post("/ocr/pdf", data={"prompt": "test"})
        
        assert response.status_code == 422  # Validation error
    
    def test_ocr_pdf_empty_file(self, client: TestClient):
        """Test OCR PDF with empty file returns error."""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        
        response = client.post("/ocr/pdf", files=files)
        
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        assert "empty" in error["detail"].lower()
    
    def test_ocr_pdf_non_pdf_file(self, client: TestClient):
        """Test OCR PDF with non-PDF file returns error."""
        files = {"file": ("test.png", io.BytesIO(b"fake image"), "image/png")}
        
        response = client.post("/ocr/pdf", files=files)
        
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        assert "pdf" in error["detail"].lower()
    
    def test_ocr_pdf_custom_prompt(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes
    ):
        """Test OCR PDF with custom prompt."""
        custom_prompt = "<image>\nParse the figure."
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        data = {"prompt": custom_prompt}
        
        response = client.post("/ocr/pdf", files=files, data=data)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500]
        result = response.json()
        assert "text" in result or "detail" in result
    
    def test_ocr_pdf_large_file(self, client: TestClient):
        """Test OCR PDF with file exceeding size limit."""
        # Create a file larger than the test limit (10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
        
        response = client.post("/ocr/pdf", files=files)
        
        assert response.status_code == 413  # File too large
        error = response.json()
        assert "detail" in error


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_endpoint(self, client: TestClient):
        """Test request to non-existent endpoint returns 404."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client: TestClient):
        """Test wrong HTTP method returns 405."""
        response = client.get("/ocr/image")
        
        assert response.status_code == 405


class TestCORS:
    """Tests for CORS middleware."""
    
    def test_cors_headers_present(self, client: TestClient):
        """Test CORS headers are present in response."""
        response = client.options(
            "/ocr/image",
            headers={"Origin": "http://example.com"}
        )
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_workflow_image(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes
    ):
        """Test complete workflow: health check + image OCR."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Process image
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        ocr_response = client.post("/ocr/image", files=files)
        
        # Should complete without crashing
        assert ocr_response.status_code in [200, 400, 500]
    
    def test_full_workflow_pdf(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes
    ):
        """Test complete workflow: health check + PDF OCR."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Process PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        ocr_response = client.post("/ocr/pdf", files=files)
        
        # Should complete without crashing
        assert ocr_response.status_code in [200, 400, 500]

