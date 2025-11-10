# -*- coding: utf-8 -*-
"""
DeepSeek-OCR API - Main entry point.

This is a simple entry point that initializes the application
and starts the server.
"""
import sys
from typing import NoReturn
import uvicorn

from config import get_config
from logger import get_logger
from services import get_ocr_service
from api import create_app
from constants import (
    LOG_STARTING_API,
    LOG_SERVER_STARTING,
    LOG_SERVER_STOPPED,
    LOG_SERVER_START_FAILED,
    COMPONENT_STARTUP,
    COMPONENT_SHUTDOWN,
)


def main() -> None:
    """
    Main entry point for the DeepSeek-OCR API.
    
    This function:
    1. Loads configuration from environment variables
    2. Initializes the logger
    3. Loads the OCR model
    4. Creates the FastAPI application
    5. Starts the uvicorn server
    """
    # Load configuration
    config = get_config()
    logger = get_logger(version=config.version)
    
    # Log startup
    logger.info(
        LOG_STARTING_API,
        component=COMPONENT_STARTUP,
        version=config.version,
        port=config.port,
        host=config.host
    )
    
    try:
        # Initialize OCR service and load model
        ocr_service = get_ocr_service(config=config)
        ocr_service.load_model()
        
        # Create FastAPI app
        app = create_app(config=config, ocr_service=ocr_service)
        
        # Start server
        logger.info(
            LOG_SERVER_STARTING.format(host=config.host, port=config.port),
            component=COMPONENT_STARTUP,
            host=config.host,
            port=config.port
        )
        
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info(LOG_SERVER_STOPPED, component=COMPONENT_SHUTDOWN)
        sys.exit(0)
    except Exception as e:
        logger.critical(
            LOG_SERVER_START_FAILED,
            component=COMPONENT_STARTUP,
            exc_info=e
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

