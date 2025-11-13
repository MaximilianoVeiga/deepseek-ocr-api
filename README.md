# DeepSeek-OCR API

> A high-performance FastAPI service for OCR processing using the DeepSeek-OCR model with GPU acceleration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-required-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Local Development](#local-development)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Image OCR](#image-ocr)
  - [PDF OCR](#pdf-ocr)
  - [Python Examples](#python-examples)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Contributing](#-contributing)
- [License](#-license)

## Overview

DeepSeek-OCR API is a production-ready REST API built with FastAPI that leverages the powerful DeepSeek-OCR model for document processing. It supports image and PDF processing with GPU acceleration, converting documents to structured markdown format.

The API uses HuggingFace's `transformers` library and provides a simple HTTP interface for OCR tasks with features like health checks, detailed error handling, and comprehensive logging.

## Features

- **High Performance** - GPU-accelerated processing with CUDA support
- **Multiple Formats** - Support for images (PNG, JPG, WEBP, BMP, TIFF) and multi-page PDFs
- **Flexible Prompts** - Customizable prompts for different OCR scenarios
- **Structured Output** - Convert documents to clean Markdown format
- **Health Monitoring** - Built-in health check endpoints with system information
- **Interactive Docs** - Auto-generated Swagger/OpenAPI documentation
- **Docker Ready** - Easy deployment with Docker and Docker Compose
- **Type Safe** - Full type hints and Pydantic validation
- **Comprehensive Logging** - Structured logging with configurable levels
- **Async Processing** - Asynchronous request handling for better performance

## Requirements

### Minimum Hardware
- NVIDIA GPU with CUDA support (A100, RTX 3090, RTX 4090, or similar)
- 16GB+ VRAM (recommended)
- 32GB+ RAM

### Software
- **NVIDIA Drivers** - Latest stable version
- **CUDA 11.8+** - Required for GPU acceleration
- **Docker 20.10+** - For containerized deployment
- **Docker Compose v2.0+** - For orchestration
- **nvidia-container-toolkit** - For GPU access in Docker
- **Python 3.10+** - For local development

### Python Dependencies
All Python dependencies are managed via `pyproject.toml`:
- FastAPI 0.115.5
- PyTorch 2.0.0+
- Transformers 4.47.1
- Flash-Attention 2.7.3+ (for optimized inference)
- See `pyproject.toml` for complete list

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deepseek-ocr-api.git
cd deepseek-ocr-api

# 2. Copy and configure environment variables
cp env.example .env

# 3. Start with Docker
docker compose up --build
```

The API will be available at `http://localhost:3000`

**Test it:**
```bash
curl http://localhost:3000/health
```

## Installation

### Docker (Recommended)

Docker provides the easiest and most reliable way to run the API with all dependencies correctly configured.

#### Prerequisites
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop) or Docker Engine
2. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Verify GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

#### Steps

1. **Clone and configure:**
   ```bash
   git clone https://github.com/yourusername/deepseek-ocr-api.git
   cd deepseek-ocr-api
   cp env.example .env
   ```

2. **Edit `.env` file** (optional):
   ```bash
   # Adjust settings as needed
   PORT=3000
   MAX_FILE_SIZE_MB=50
   ```

3. **Build and start:**
   ```bash
   docker compose up --build
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:3000/health/detailed
   ```

### Local Development

For development or when Docker is not available, you can run the API locally.

#### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+
- pip or uv package manager

#### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/yourusername/deepseek-ocr-api.git
cd deepseek-ocr-api

# Copy environment file
Copy-Item env.example .env

# Install dependencies (first time)
.\scripts\install-deps.ps1

# Start server
.\scripts\start-server.ps1
```

#### Linux/macOS

```bash
# Clone repository
git clone https://github.com/yourusername/deepseek-ocr-api.git
cd deepseek-ocr-api

# Copy environment file
cp env.example .env

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -e .
# Or for development:
pip install -e ".[dev]"

# Start server
python main.py
```

## Configuration

Configuration is managed through environment variables. Copy `env.example` to `.env` and adjust as needed.

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3000` | Server port |
| `ENVIRONMENT` | `development` | Environment mode (`development`/`production`) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DS_OCR_MODEL` | `unsloth/DeepSeek-OCR` | HuggingFace model identifier |
| `DEVICE` | `auto` | Device for inference (`auto`, `cuda`, `mps`, `cpu`) |
| `BASE_SIZE` | `1024` | Base size for model processing |
| `IMAGE_SIZE` | `640` | Image preprocessing size |

### OCR Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_PROMPT` | `<image>\n<\|grounding\|>Convert the document to markdown format with proper headers, lists, tables, and formatting.` | Default OCR prompt |
| `PDF_DPI` | `220` | DPI for PDF to image conversion |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |
| `MAX_PDF_PAGES` | `100` | Maximum pages per PDF |

### CORS Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY_ENABLED` | `false` | Enable API key authentication |
| `API_KEY` | `""` | API key for authentication (required if enabled) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Maximum requests per minute per IP |
| `PROMPT_MAX_LENGTH` | `2000` | Maximum length for OCR prompts |

## Usage

### Image OCR

Process images and extract text as Markdown:

```bash
curl -X POST http://localhost:3000/ocr/image \
  -F "file=@invoice.jpg" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown."
```

**Response:**
```json
{
  "success": true,
  "text": "# Invoice\n\n**Invoice Number:** INV-001...",
  "filename": "invoice.jpg",
  "processing_time_seconds": 2.35,
  "model_version": "unsloth/DeepSeek-OCR",
  "correlation_id": "abc123-def456"
}
```

#### Supported Image Formats
- PNG
- JPEG/JPG
- WEBP
- BMP
- TIFF

### PDF OCR

Process multi-page PDFs:

```bash
curl -X POST http://localhost:3000/ocr/pdf \
  -F "file=@contract.pdf" \
  -F "dpi=220"
```

**Response:**
```json
{
  "success": true,
  "pages": [
    {
      "page_number": 1,
      "text": "# Contract Agreement\n\n**Date:** 2024-01-15...",
      "processing_time_seconds": 2.10,
      "success": true
    },
    {
      "page_number": 2,
      "text": "## Terms and Conditions\n\n1. Payment terms...",
      "processing_time_seconds": 2.05,
      "success": true
    }
  ],
  "total_pages": 2,
  "filename": "contract.pdf",
  "total_processing_time_seconds": 4.15,
  "model_version": "unsloth/DeepSeek-OCR",
  "correlation_id": "xyz789-abc123",
  "warnings": []
}
```

### Python Examples

#### Using requests library

```python
import requests

# Image OCR
with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:3000/ocr/image",
        files={"file": f},
        data={"prompt": "<image>\n<|grounding|>Convert the document to markdown."}
    )
    
result = response.json()
print(result["text"])

# PDF OCR
with open("report.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:3000/ocr/pdf",
        files={"file": f},
        data={"dpi": 220}
    )
    
result = response.json()
for page in result["pages"]:
    print(f"Page {page['page_number']}:")
    print(page["text"])
    print("-" * 50)
```

#### Using httpx (async)

```python
import httpx
import asyncio

async def process_document():
    async with httpx.AsyncClient() as client:
        with open("document.jpg", "rb") as f:
            response = await client.post(
                "http://localhost:3000/ocr/image",
                files={"file": ("document.jpg", f, "image/jpeg")},
                data={"prompt": "<image>\n<|grounding|>Convert the document to markdown."}
            )
        
        result = response.json()
        return result["text"]

# Run
text = asyncio.run(process_document())
print(text)
```

### Prompt Examples

Different prompts for different use cases:

#### Document to Markdown (Default)
```
<image>
<|grounding|>Convert the document to markdown format with proper headers, lists, tables, and formatting.
```

#### Simple OCR (No Layout)
```
<image>
Free OCR.
```

#### Figure Parsing
```
<image>
Parse the figure.
```

#### Entity Localization
```
<image>
Locate <|ref|>IBAN<|/ref|> in the image.
```

#### Table Extraction
```
<image>
<|grounding|>Extract all tables from this document.
```

## API Documentation

The API provides interactive documentation powered by FastAPI:

### Available Endpoints

#### Health Check
- **`GET /health`** - Basic health check
  ```bash
  curl http://localhost:3000/health
  ```

#### OCR Processing
- **`POST /ocr/image`** - Process image files
- **`POST /ocr/pdf`** - Process PDF files

#### Kubernetes Health Probes
- **`GET /health/ready`** - Readiness probe (returns 200 when model is loaded)
- **`GET /health/live`** - Liveness probe (returns 200 when service is responsive)

### Interactive Documentation

With the server running, access:

- **Swagger UI (Interactive):** http://localhost:3000/docs
- **ReDoc (Alternative view):** http://localhost:3000/redoc
- **OpenAPI JSON Schema:** http://localhost:3000/openapi.json

The Swagger UI allows you to:
- Test endpoints directly from your browser
- View detailed request/response schemas
- See example payloads and responses
- Download the OpenAPI specification

## Project Structure

```
deepseek-ocr-api/
├── api/                          # FastAPI application
│   ├── __init__.py              # API package initialization
│   ├── app.py                   # FastAPI app factory
│   ├── dependencies.py          # Dependency injection
│   ├── handlers.py              # Exception handlers
│   ├── middleware.py            # CORS and other middleware
│   ├── routes.py                # Route configuration
│   └── routers/                 # API route modules
│       ├── health.py            # Health check endpoints
│       └── ocr.py               # OCR processing endpoints
├── models/                       # Data models
│   ├── schemas.py               # Pydantic request/response models
│   ├── validators.py            # Custom validators
│   └── exceptions.py            # Custom exceptions
├── services/                     # Business logic
│   └── ocr_service.py           # OCR processing service
├── utils/                        # Utility functions
│   └── files.py                 # File handling utilities
├── scripts/                      # Helper scripts
│   ├── install-deps.ps1         # Windows dependency installer
│   ├── start-server.ps1         # Windows server starter
│   └── run-tests.ps1            # Windows test runner
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest configuration
│   ├── test_api.py              # API endpoint tests
│   └── test_ocr_service.py      # Service layer tests
├── config.py                     # Configuration management
├── constants.py                  # Application constants
├── logger.py                     # Logging configuration
├── main.py                       # Application entry point
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # Pip requirements (generated)
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── env.example                  # Example environment variables
└── README.md                    # This file
```

## Development

### Setting Up Development Environment

1. **Clone and install:**
   ```bash
   git clone https://github.com/yourusername/deepseek-ocr-api.git
   cd deepseek-ocr-api
   pip install -e ".[dev]"
   ```

2. **Configure pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Run in development mode:**
   ```bash
   python main.py
   ```

### Code Style

The project follows:
- PEP 8 style guidelines
- Type hints throughout
- Docstrings for all public functions
- Async/await patterns for I/O operations

### Making Changes

1. Create a feature branch
2. Make your changes
3. Add/update tests
4. Run the test suite
5. Submit a pull request

## Testing

The project includes a comprehensive test suite using pytest.

### Run All Tests

```bash
# With pytest
pytest

# With coverage
pytest --cov=. --cov-report=html

# Windows PowerShell
.\scripts\run-tests.ps1
```

### Run Specific Tests

```bash
# Test API endpoints only
pytest tests/test_api.py

# Test OCR service only
pytest tests/test_ocr_service.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_api.py::test_health_check
```

### Test Categories

- **Unit Tests** - Test individual components
- **Integration Tests** - Test component interactions
- **API Tests** - Test HTTP endpoints

Use markers to run specific categories:
```bash
# Skip integration tests
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```

## Troubleshooting

### Common Issues

#### GPU Not Detected

**Symptoms:** Model runs on CPU, very slow processing

**Solutions:**
1. Check CUDA installation:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Verify nvidia-container-toolkit (Docker):
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Set `DEVICE=cuda` explicitly in `.env`

#### Out of Memory (OOM)

**Symptoms:** CUDA out of memory errors

**Solutions:**
1. Reduce batch size parameters in model config
2. Lower `IMAGE_SIZE` and `BASE_SIZE` in `.env`
3. Process smaller files or reduce `PDF_DPI`
4. Close other GPU-using applications

#### Model Download Issues

**Symptoms:** Model fails to download from HuggingFace

**Solutions:**
1. Check internet connection
2. Set HuggingFace token if model requires authentication:
   ```bash
   export HF_TOKEN=your_token_here
   ```
3. Manually download model:
   ```python
   from transformers import AutoModel
   AutoModel.from_pretrained("unsloth/DeepSeek-OCR")
   ```

#### Port Already in Use

**Symptoms:** `Address already in use` error

**Solutions:**
1. Change port in `.env`:
   ```bash
   PORT=3001
   ```
2. Or kill the process using the port:
   ```bash
   # Linux/Mac
   lsof -ti:3000 | xargs kill -9
   
   # Windows
   netstat -ano | findstr :3000
   taskkill /PID <PID> /F
   ```

#### Slow Processing

**Symptoms:** OCR takes very long time

**Solutions:**
1. Verify GPU is being used (check logs)
2. Reduce image size: lower `IMAGE_SIZE` in `.env`
3. Use lower DPI for PDFs: `dpi=150` instead of `220`
4. Ensure Flash-Attention is properly installed
5. Check GPU utilization: `nvidia-smi -l 1`

### Debug Mode

Enable detailed logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or export before running
export LOG_LEVEL=DEBUG
python main.py
```

### Getting Help

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review API logs for error messages
3. Search existing [GitHub Issues](https://github.com/yourusername/deepseek-ocr-api/issues)
4. Create a new issue with:
   - System information (`GET /health/detailed`)
   - Error messages and logs
   - Steps to reproduce

## Performance Tips

### Optimizing OCR Processing

1. **Image Preprocessing**
   - Use appropriate DPI: 150-300 for most documents
   - Ensure images are properly oriented
   - Remove unnecessary margins/borders

2. **Model Configuration**
   - Adjust `BASE_SIZE` based on document complexity
   - Lower `IMAGE_SIZE` for faster processing of simple documents
   - Use Flash-Attention 2 for 2-4x speedup

3. **Hardware Optimization**
   - Use GPU with at least 16GB VRAM
   - Ensure proper cooling for sustained performance
   - Close unnecessary GPU-using applications

4. **Batch Processing**
   - Process multiple files sequentially rather than parallel
   - Use appropriate file size limits
   - Implement queueing for high-volume scenarios

### Production Deployment

For production environments:

1. **Security**
   - Add API key authentication
   - Set `CORS_ORIGINS` to specific domains
   - Use HTTPS with reverse proxy (nginx/traefik)
   - Implement rate limiting

2. **Scalability**
   - Deploy behind load balancer
   - Implement job queue (Celery/Redis)
   - Use horizontal scaling for multiple GPU instances
   - Add caching layer for common requests

3. **Monitoring**
   - Set up application monitoring (Prometheus/Grafana)
   - Track GPU metrics and utilization
   - Monitor API response times
   - Set up alerts for errors and downtime

4. **Reliability**
   - Implement retries with exponential backoff
   - Add circuit breakers for external dependencies
   - Use persistent logging (ELK stack, CloudWatch)
   - Regular health checks and auto-restart

## Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

1. Check if the bug is already reported
2. Include detailed reproduction steps
3. Provide system information and logs
4. Use the bug report template

### Suggesting Features

1. Check existing feature requests
2. Clearly describe the use case
3. Explain expected behavior
4. Use the feature request template

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add/update tests
5. Ensure all tests pass
6. Update documentation
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Guidelines

- Follow existing code style
- Add type hints
- Write docstrings
- Update tests
- Keep commits atomic
- Write clear commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai) for the OCR model
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [HuggingFace](https://huggingface.co/) for the transformers library
- All contributors and users of this project

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/deepseek-ocr-api/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/deepseek-ocr-api/discussions)
- **Author:** Maximiliano Veiga

---

** If you find this project useful, please consider giving it a star on GitHub!**
