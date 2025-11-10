# -*- coding: utf-8 -*-
"""Logging utilities for DeepSeek-OCR API."""
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional, Dict
from enum import Enum


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured logger with consistent formatting.
    
    Logs include: component, version, timestamp, level, and message.
    Format: [DeepSeek-OCR v{VERSION}] {TIMESTAMP} {LEVEL} [{COMPONENT}] {MESSAGE}
    """
    
    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the structured logger.
        
        Args:
            version: Application version
        """
        self.version = version
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("deepseek-ocr")
    
    def _format_message(
        self,
        level: LogLevel,
        message: str,
        component: str = "main",
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Format log message with consistent structure.
        
        Args:
            level: Log level
            message: Log message
            component: Component name
            **kwargs: Additional context to include
            
        Returns:
            Formatted log message
        """
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        base_msg = f"[DeepSeek-OCR v{self.version}] {timestamp} {level.value} [{component}] {message}"
        
        if kwargs:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{base_msg} | {context}"
        
        return base_msg
    
    def debug(
        self,
        message: str,
        component: str = "main",
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Log debug message.
        
        Args:
            message: Log message
            component: Component name
            **kwargs: Additional context
        """
        formatted = self._format_message(LogLevel.DEBUG, message, component, **kwargs)
        self.logger.debug(formatted)
    
    def info(
        self,
        message: str,
        component: str = "main",
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Log info message.
        
        Args:
            message: Log message
            component: Component name
            **kwargs: Additional context
        """
        formatted = self._format_message(LogLevel.INFO, message, component, **kwargs)
        self.logger.info(formatted)
    
    def warning(
        self,
        message: str,
        component: str = "main",
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Log warning message.
        
        Args:
            message: Log message
            component: Component name
            **kwargs: Additional context
        """
        formatted = self._format_message(LogLevel.WARNING, message, component, **kwargs)
        self.logger.warning(formatted)
    
    def error(
        self,
        message: str,
        component: str = "main",
        exc_info: Optional[Exception] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Log error message.
        
        Args:
            message: Log message
            component: Component name
            exc_info: Exception information
            **kwargs: Additional context
        """
        if exc_info:
            kwargs["error"] = str(exc_info)
            kwargs["error_type"] = type(exc_info).__name__
        
        formatted = self._format_message(LogLevel.ERROR, message, component, **kwargs)
        self.logger.error(formatted, exc_info=exc_info is not None)
    
    def critical(
        self,
        message: str,
        component: str = "main",
        exc_info: Optional[Exception] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Log critical message.
        
        Args:
            message: Log message
            component: Component name
            exc_info: Exception information
            **kwargs: Additional context
        """
        if exc_info:
            kwargs["error"] = str(exc_info)
            kwargs["error_type"] = type(exc_info).__name__
        
        formatted = self._format_message(LogLevel.CRITICAL, message, component, **kwargs)
        self.logger.critical(formatted, exc_info=exc_info is not None)


def get_logger(version: str = "1.0.0") -> StructuredLogger:
    """
    Create and return a logger instance.
    
    Note: For dependency injection, logger is stored in app state.
    This function is used for initialization.
    
    Args:
        version: Application version
        
    Returns:
        StructuredLogger: Logger instance
    """
    return StructuredLogger(version=version)

