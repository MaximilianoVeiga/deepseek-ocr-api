# -*- coding: utf-8 -*-
"""Model loading and lifecycle management."""
import warnings
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import torch

from config import Config
from logger import StructuredLogger
from models import OCRProcessingError
from constants import (
    ERROR_MODEL_LOAD_FAILED,
    LOG_MODEL_LOADING,
    LOG_MODEL_LOADED,
    LOG_MODEL_ALREADY_LOADED,
    LOG_MODEL_LOAD_FAILED,
    COMPONENT_STARTUP,
    COMPONENT_OCR_SERVICE,
    DEVICE_CUDA,
    DEVICE_CPU,
)


class ModelLoader:
    """
    Handles model and tokenizer loading with device placement.
    
    Manages the lifecycle of the DeepSeek-OCR model including:
    - Loading model and tokenizer from HuggingFace
    - Device placement (CUDA/CPU/MPS)
    - Warning suppression during loading
    - Model state tracking
    """
    
    def __init__(self, config: Config, logger: StructuredLogger):
        """
        Initialize model loader.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """
        Load the DeepSeek-OCR model and tokenizer.
        
        Raises:
            OCRProcessingError: If model loading fails
        """
        if self._model_loaded:
            self.logger.info(
                LOG_MODEL_ALREADY_LOADED,
                component=COMPONENT_OCR_SERVICE
            )
            return
        
        try:
            self.logger.info(
                LOG_MODEL_LOADING.format(model_name=self.config.model_name),
                component=COMPONENT_STARTUP,
                model=self.config.model_name,
                device=self.config.device
            )
            
            # Suppress expected warnings during model loading
            warnings.filterwarnings("ignore", message="You are using a model of type")
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
            warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
            warnings.filterwarnings("ignore", message="You should probably TRAIN this model.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                
                # Configure tokenizer padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.logger.info(
                        "Tokenizer pad_token not set, using eos_token as pad_token",
                        component=COMPONENT_STARTUP
                    )
                
                # Load model and move to appropriate device
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
            finally:
                # Reset warning filters
                warnings.resetwarnings()
            
            # Move model to device
            if self.config.device == DEVICE_CUDA:
                self.model = self.model.cuda().to(torch.bfloat16)
            elif self.config.device == DEVICE_CPU:
                self.model = self.model.cpu()
            else:
                # MPS or other devices
                self.model = self.model.to(self.config.device)
            
            self._model_loaded = True
            self.logger.info(
                LOG_MODEL_LOADED,
                component=COMPONENT_STARTUP,
                model=self.config.model_name,
                device=self.config.device
            )
            
        except Exception as e:
            self.logger.error(
                LOG_MODEL_LOAD_FAILED,
                component=COMPONENT_STARTUP,
                exc_info=e,
                model=self.config.model_name
            )
            raise OCRProcessingError(
                ERROR_MODEL_LOAD_FAILED.format(error=str(e)),
                original_error=e
            )
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self._model_loaded and self.model is not None and self.tokenizer is not None
    
    def get_model(self) -> AutoModel:
        """
        Get the loaded model.
        
        Returns:
            AutoModel: The loaded model
        """
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get the loaded tokenizer.
        
        Returns:
            AutoTokenizer: The loaded tokenizer
        """
        return self.tokenizer

