# -*- coding: utf-8 -*-
"""Pytest configuration and fixtures for DeepSeek-OCR API tests."""
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from services import OCRService
from api import create_app


@pytest.fixture(scope="session")
def test_config() -> Config:
    """
    Create test configuration.
    
    Returns:
        Config: Test configuration with reduced limits
    """
    return Config(
        host="127.0.0.1",
        port=3000,
        model_name=os.getenv("DS_OCR_MODEL", "deepseek-ai/DeepSeek-OCR"),
        max_file_size_mb=10,  # Smaller limit for tests
        max_pdf_pages=10,      # Smaller limit for tests
        version="1.0.0-test"
    )


@pytest.fixture(scope="session")
def ocr_service(test_config: Config) -> Generator[OCRService, None, None]:
    """
    Create OCR service for testing.
    
    This fixture loads the model once for all tests.
    
    Args:
        test_config: Test configuration
        
    Yields:
        OCRService: Initialized OCR service with loaded model
    """
    service = OCRService(config=test_config)
    service.load_model()
    yield service


@pytest.fixture(scope="session")
def app(test_config: Config, ocr_service: OCRService):
    """
    Create FastAPI application for testing.
    
    Args:
        test_config: Test configuration
        ocr_service: OCR service
        
    Returns:
        FastAPI: Application instance
    """
    return create_app(config=test_config, ocr_service=ocr_service)


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """
    Create test client for synchronous tests.
    
    Args:
        app: FastAPI application
        
    Yields:
        TestClient: Test client
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async test client for async tests.
    
    Args:
        app: FastAPI application
        
    Yields:
        AsyncClient: Async test client
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """
    Get path to test fixtures directory.
    
    Returns:
        Path: Path to fixtures directory
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """
    Get path to examples directory (fallback for test files).
    
    Returns:
        Path: Path to examples directory
    """
    return Path(__file__).parent.parent.parent / "examples"


@pytest.fixture
def test_image_path(examples_dir: Path) -> Path:
    """
    Get path to a test image file.
    
    Uses examples directory as fixtures.
    
    Args:
        examples_dir: Examples directory path
        
    Returns:
        Path: Path to test image
    """
    # Use one of the PDF files from examples (PDFs can be converted to images)
    # For actual image tests, you'd want to create proper test images
    return examples_dir / "T1.pdf"


@pytest.fixture
def test_pdf_path(examples_dir: Path) -> Path:
    """
    Get path to a test PDF file.
    
    Uses examples directory as fixtures.
    
    Args:
        examples_dir: Examples directory path
        
    Returns:
        Path: Path to test PDF
    """
    return examples_dir / "P1.pdf"


@pytest.fixture
def sample_image_bytes(test_image_path: Path) -> bytes:
    """
    Read sample image file as bytes.
    
    Args:
        test_image_path: Path to test image
        
    Returns:
        bytes: Image file content
    """
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")
    return test_image_path.read_bytes()


@pytest.fixture
def sample_pdf_bytes(test_pdf_path: Path) -> bytes:
    """
    Read sample PDF file as bytes.
    
    Args:
        test_pdf_path: Path to test PDF
        
    Returns:
        bytes: PDF file content
    """
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found: {test_pdf_path}")
    return test_pdf_path.read_bytes()


@pytest.fixture
def mock_prompt() -> str:
    """
    Get a mock OCR prompt for testing.
    
    Returns:
        str: Test prompt
    """
    return "<image>\n<|grounding|>Convert the document to markdown."


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires GPU)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

