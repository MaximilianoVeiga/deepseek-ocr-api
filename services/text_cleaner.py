# -*- coding: utf-8 -*-
"""Text processing and cleaning utilities."""
import re
from typing import Optional

from logger import StructuredLogger
from constants import COMPONENT_OCR_SERVICE


class TextCleaner:
    """
    Handles text processing and cleaning operations.
    
    Provides utilities for:
    - Stripping grounding annotations
    - Cleaning captured stdout
    - Text normalization
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize text cleaner.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
    
    def strip_grounding_annotations(self, text: str) -> str:
        """
        Strip grounding annotations from OCR output.
        
        Removes reference tags (<|ref|>...<|/ref|>) and detection bounding boxes
        (<|det|>[[...]]<|/det|>) to produce clean text output.
        
        Args:
            text: Text with grounding annotations
            
        Returns:
            str: Clean text without annotations
        """
        if not text:
            return ""
        
        # Remove reference tags: <|ref|>type<|/ref|>
        text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
        
        # Remove detection bounding boxes: <|det|>[[x1, y1, x2, y2]]<|/det|>
        text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
        
        # Clean up multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up spaces at the start of lines
        lines = text.split('\n')
        cleaned_lines = [line.strip() if line.strip() else '' for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_stdout_output(self, stdout_text: str, strip_grounding: bool = True) -> str:
        """
        Clean captured stdout to extract OCR text.
        
        Filters out debug messages and extracts the actual OCR content.
        Optionally strips grounding annotations for clean output.
        
        Args:
            stdout_text: Raw captured stdout
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            str: Cleaned OCR text
        """
        if not stdout_text:
            return ""
        
        lines = stdout_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip debug lines and noise
            if any([
                line.startswith('====='),
                line.startswith('BASE:'),
                line.startswith('PATCHES:'),
                line.strip() == '(0x0)',
                line.strip().startswith('(0x0)') and len(line.strip()) < 50,
                line.strip() == '0x0',
                'torch.Size' in line,
            ]):
                continue
            
            # Keep actual content
            if line.strip():
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # Strip grounding annotations if requested
        if strip_grounding:
            result = self.strip_grounding_annotations(result)
        
        return result

