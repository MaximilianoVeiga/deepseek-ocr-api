# -*- coding: utf-8 -*-
"""File handling utilities."""
import os
import tempfile
from contextlib import contextmanager
from typing import Generator

from constants import TEMP_FILE_PREFIX
from logger import StructuredLogger


@contextmanager
def temporary_file(
    suffix: str = "",
    prefix: str = TEMP_FILE_PREFIX,
    logger: StructuredLogger | None = None
) -> Generator[str, None, None]:
    """
    Context manager for temporary file handling.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        logger: Optional logger for cleanup warnings
        
    Yields:
        str: Path to temporary file
    """
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=prefix
    )
    tmp_path = tmp_file.name
    tmp_file.close()
    
    try:
        yield tmp_path
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            if logger:
                from constants import LOG_TEMP_FILE_CLEANUP_FAILED, COMPONENT_OCR_SERVICE
                logger.warning(
                    LOG_TEMP_FILE_CLEANUP_FAILED.format(path=tmp_path),
                    component=COMPONENT_OCR_SERVICE,
                    exc_info=e
                )

