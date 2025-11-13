# -*- coding: utf-8 -*-
"""Core model inference execution."""
import os
import contextlib
import io
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Optional, Any, Tuple
from transformers import AutoModel, AutoTokenizer

from config import Config
from logger import StructuredLogger
from models import OCRProcessingError
from constants import (
    ERROR_IMAGE_INFERENCE_FAILED,
    COMPONENT_OCR_SERVICE,
)
from .text_cleaner import TextCleaner


class InferenceEngine:
    """
    Handles core model inference execution.
    
    Manages:
    - Running model inference with captured stdout
    - Extracting text from model results
    - Handling output files
    - Temporary directory management
    """
    
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
        text_cleaner: Optional[TextCleaner] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object
            logger: Logger instance
            text_cleaner: Optional text cleaner instance
        """
        self.config = config
        self.logger = logger
        self.text_cleaner = text_cleaner or TextCleaner(logger)
    
    def run_model_inference(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        image_path: str,
        prompt: str,
        output_path: str,
        save_results: bool = False
    ) -> Tuple[Any, str]:
        """
        Run model inference and capture output.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer to use
            image_path: Path to image file
            prompt: OCR prompt
            output_path: Path for model output files
            save_results: Whether to save results to file
            
        Returns:
            tuple: (model result, captured stdout)
        """
        stdout_capture = io.StringIO()
        result = None
        
        # Suppress expected warnings during inference
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*pad token.*")
        warnings.filterwarnings("ignore", message="Setting `pad_token_id`.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=output_path,
                    base_size=self.config.base_size,
                    image_size=self.config.image_size,
                    crop_mode=True,
                    save_results=save_results,
                    test_compress=False,
                )
        finally:
            # Reset warning filters
            warnings.resetwarnings()
        
        stdout_text = stdout_capture.getvalue()
        return result, stdout_text
    
    def extract_text_from_result(self, result: Any) -> Optional[str]:
        """
        Extract text from model inference result.
        
        Args:
            result: Model inference result
            
        Returns:
            Optional[str]: Extracted text, or None if not found
        """
        if isinstance(result, str) and result:
            return result
        elif isinstance(result, dict):
            # Check for common keys that might contain the text
            for key in ['text', 'output', 'result', 'prediction']:
                if key in result and result[key]:
                    return str(result[key])
            self.logger.warning(
                f"Dict returned but no text found. Keys: {list(result.keys())}",
                component=COMPONENT_OCR_SERVICE
            )
        elif isinstance(result, list) and result:
            # If list, concatenate all string elements
            text = "\n".join(str(item) for item in result if item)
            if text:
                return text
        return None
    
    def try_read_output_files(
        self,
        output_path: str,
        strip_grounding: bool = True
    ) -> Optional[str]:
        """
        Try to read OCR results from output files.
        
        Args:
            output_path: Directory containing output files
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            Optional[str]: Text from output files, or None if not found
        """
        output_files = list(Path(output_path).glob("*.txt")) + \
                      list(Path(output_path).glob("*.md"))
        
        if output_files:
            output_file = output_files[0]
            self.logger.info(
                f"Reading result from file: {output_file}",
                component=COMPONENT_OCR_SERVICE
            )
            with open(output_file, "r", encoding="utf-8") as f:
                text = f.read()
                # Apply grounding stripping if requested
                if strip_grounding:
                    text = self.text_cleaner.strip_grounding_annotations(text)
                return text
        return None
    
    def infer_image(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        image_path: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> str:
        """
        Perform OCR inference on a single image.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer to use
            image_path: Path to image file
            prompt: OCR prompt
            strip_grounding: Whether to strip grounding annotations (default: True)
            
        Returns:
            str: Extracted text
            
        Raises:
            OCRProcessingError: If inference fails
        """
        try:
            # Create a temporary directory for model output
            temp_output_dir = tempfile.mkdtemp(prefix="dsocr-output-")
            
            try:
                self.logger.info(
                    f"Calling model.infer with prompt: {prompt[:100]}...",
                    component=COMPONENT_OCR_SERVICE
                )
                
                # Run model inference
                result, stdout_text = self.run_model_inference(
                    model, tokenizer, image_path, prompt, temp_output_dir, save_results=False
                )
                
                # Log what we got back (DEBUG level - model typically returns None and uses stdout)
                self.logger.debug(
                    f"Model inference complete: result_type={type(result).__name__}, stdout_length={len(stdout_text)}",
                    component=COMPONENT_OCR_SERVICE
                )
                
                # Priority 1: Try to extract text from captured stdout (this is the expected path)
                if stdout_text:
                    cleaned_text = self.text_cleaner.clean_stdout_output(stdout_text, strip_grounding)
                    if cleaned_text:
                        self.logger.info(
                            f"Extracted text from stdout ({len(cleaned_text)} chars)",
                            component=COMPONENT_OCR_SERVICE
                        )
                        return cleaned_text
                
                # Priority 2: Check if result has text
                extracted_text = self.extract_text_from_result(result)
                if extracted_text:
                    # Apply grounding stripping if requested
                    if strip_grounding:
                        extracted_text = self.text_cleaner.strip_grounding_annotations(extracted_text)
                    return extracted_text
                
                # Priority 3: Try saving results to file as fallback
                self.logger.info(
                    "No text from stdout or result, trying with save_results=True",
                    component=COMPONENT_OCR_SERVICE
                )
                
                _, stdout_text2 = self.run_model_inference(
                    model, tokenizer, image_path, prompt, temp_output_dir, save_results=True
                )
                
                # Try stdout from second call
                if stdout_text2:
                    cleaned_text2 = self.text_cleaner.clean_stdout_output(stdout_text2, strip_grounding)
                    if cleaned_text2:
                        self.logger.info(
                            f"Extracted text from fallback stdout ({len(cleaned_text2)} chars)",
                            component=COMPONENT_OCR_SERVICE
                        )
                        return cleaned_text2
                
                # Look for output files
                file_text = self.try_read_output_files(temp_output_dir, strip_grounding)
                if file_text:
                    return file_text
                
                # Nothing worked
                self.logger.warning(
                    "Unable to extract text from model inference",
                    component=COMPONENT_OCR_SERVICE
                )
                return ""
                
            finally:
                # Clean up temporary output directory
                try:
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to clean up temporary output directory: {temp_output_dir}",
                        component=COMPONENT_OCR_SERVICE,
                        exc_info=cleanup_error
                    )
            
        except Exception as e:
            self.logger.error(
                "Image inference failed",
                component=COMPONENT_OCR_SERVICE,
                exc_info=e,
                image_path=image_path
            )
            raise OCRProcessingError(
                ERROR_IMAGE_INFERENCE_FAILED.format(error=str(e)),
                original_error=e
            )

