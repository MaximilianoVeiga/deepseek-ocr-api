# -*- coding: utf-8 -*-
"""Model loading and lifecycle management."""
import os
import tempfile
import warnings
from importlib.util import find_spec
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

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


def is_flash_attention_available() -> bool:
    """
    Check if Flash-Attention 2 package is installed and available.
    
    Returns:
        bool: True if flash-attn package is available, False otherwise
    """
    return find_spec("flash_attn") is not None


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
                
                # Determine if Flash-Attention 2 should be used
                use_flash_attention = False
                if self.config.enable_flash_attention and self.config.device == DEVICE_CUDA:
                    if is_flash_attention_available():
                        use_flash_attention = True
                        self.logger.info(
                            "Flash-Attention 2 is enabled and available - using FA2 for 2-4x speedup",
                            component=COMPONENT_STARTUP
                        )
                    else:
                        self.logger.warning(
                            "Flash-Attention 2 is enabled but flash-attn package is not installed. "
                            "Install with: pip install flash-attn --no-build-isolation. "
                            "Falling back to standard attention.",
                            component=COMPONENT_STARTUP
                        )
                elif self.config.enable_flash_attention and self.config.device != DEVICE_CUDA:
                    self.logger.info(
                        "Flash-Attention 2 requires CUDA device - skipping FA2 on non-CUDA device",
                        component=COMPONENT_STARTUP
                    )
                
                # Build model loading kwargs
                model_kwargs = {
                    "trust_remote_code": True,
                    "use_safetensors": True,
                }
                
                # Add Flash-Attention 2 if available and enabled
                if use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                # Load model and move to appropriate device
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                ).eval()
            finally:
                # Reset warning filters
                warnings.resetwarnings()
            
            # Move model to device
            if self.config.device == DEVICE_CUDA:
                self.model = self.model.cuda().to(torch.bfloat16)
                
                # Enable CUDA optimizations for RTX 4080 and similar GPUs
                torch.backends.cudnn.benchmark = True  # Cache best convolution algorithms
                torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for matmul
                torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cuDNN operations
                
                self.logger.info(
                    "CUDA optimizations enabled (cuDNN benchmark, TF32)",
                    component=COMPONENT_STARTUP
                )
                
                # Compile model with torch.compile for 20-30% speedup
                if self.config.enable_torch_compile and hasattr(torch, 'compile'):
                    self.logger.info(
                        "Compiling model with torch.compile (this may take a minute)...",
                        component=COMPONENT_STARTUP
                    )
                    self.model = torch.compile(
                        self.model,
                        mode='reduce-overhead',  # Optimize for latency
                        fullgraph=False  # Allow dynamic shapes for flexibility
                    )
                    self.logger.info(
                        "Model compilation complete",
                        component=COMPONENT_STARTUP
                    )
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
    
    def warmup_model(self) -> None:
        """
        Warm up the model with a dummy inference to pre-compile CUDA kernels.
        
        This improves performance of the first real inference by pre-allocating
        GPU memory and compiling CUDA kernels ahead of time.
        """
        if not self.is_loaded():
            self.logger.warning(
                "Cannot warm up model - model not loaded",
                component=COMPONENT_OCR_SERVICE
            )
            return
        
        self.logger.info(
            "Warming up model with dummy inference to pre-compile CUDA kernels...",
            component=COMPONENT_STARTUP
        )
        
        dummy_path = None
        try:
            # Create a small dummy image
            dummy_img = Image.new('RGB', (224, 224), color='white')
            fd, dummy_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            dummy_img.save(dummy_path, 'JPEG')
            
            # Create a temporary output directory
            temp_output_dir = tempfile.mkdtemp(prefix="dsocr-warmup-")
            
            try:
                # Run dummy inference with torch.inference_mode()
                with torch.inference_mode():
                    _ = self.model.infer(
                        self.tokenizer,
                        prompt="Warmup",
                        image_file=dummy_path,
                        output_path=temp_output_dir,
                        base_size=self.config.base_size,
                        image_size=self.config.image_size,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False
                    )
                
                self.logger.info(
                    "Model warmup complete - CUDA kernels pre-compiled",
                    component=COMPONENT_STARTUP
                )
            finally:
                # Clean up temp output directory
                import shutil
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
        
        except Exception as e:
            self.logger.warning(
                f"Model warmup failed (non-critical): {str(e)}",
                component=COMPONENT_STARTUP,
                exc_info=e
            )
        finally:
            # Clean up dummy image
            if dummy_path and os.path.exists(dummy_path):
                try:
                    os.unlink(dummy_path)
                except Exception:
                    pass

