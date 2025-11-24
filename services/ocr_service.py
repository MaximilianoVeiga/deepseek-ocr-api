# -*- coding: utf-8 -*-
"""OCR service with model inference logic."""
from typing import Optional, Tuple, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass, field

from config import Config, get_config
from logger import get_logger
from models import OCRProcessingError
from constants import ERROR_MODEL_NOT_LOADED, COMPONENT_OCR_SERVICE

from .model_loader import ModelLoader
from .text_cleaner import TextCleaner
from .inference_engine import InferenceEngine
from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor


@dataclass
class JobRequest:
    """Job request for the processing queue."""
    future: asyncio.Future
    func: Callable
    args: Tuple
    kwargs: Dict = field(default_factory=dict)


class OCRService:
    """
    OCR service for processing images and PDFs with DeepSeek-OCR.
    
    This service handles model loading, inference, and document processing.
    Uses a job queue and background worker to strictly serialize GPU operations
    and manage load.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize OCR service.
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.logger = get_logger(version=self.config.version)
        
        # Thread pool for running synchronous model inference
        # Use 1 worker to avoid GPU concurrency issues
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-inference")
        
        # Job queue for serialization
        # Max size controls backpressure (reject requests if queue full)
        # Default to 100 items to allow bursts but prevent OOM
        self.queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        
        # Initialize components
        self.model_loader = ModelLoader(self.config, self.logger)
        self.text_cleaner = TextCleaner(self.logger)
        self.inference_engine = InferenceEngine(
            self.config,
            self.logger,
            self.text_cleaner
        )
        self.image_processor = ImageProcessor(
            self.config,
            self.logger,
            self.inference_engine,
            self._executor
        )
        self.pdf_processor = PDFProcessor(
            self.config,
            self.logger,
            self.inference_engine,
            self._executor
        )
    
    async def start(self) -> None:
        """
        Start the background worker.
        
        Must be called when the async event loop is running (e.g. FastAPI startup).
        """
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=100)
            
        if self._worker_task is None or self._worker_task.done():
            self.logger.info(
                "Starting OCR service worker",
                component=COMPONENT_OCR_SERVICE
            )
            self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self) -> None:
        """
        Stop the background worker and cleanup resources.
        """
        if self._worker_task and not self._worker_task.done():
            self.logger.info(
                "Stopping OCR service worker",
                component=COMPONENT_OCR_SERVICE
            )
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            
        self.shutdown()

    async def _worker_loop(self) -> None:
        """
        Background worker loop that consumes jobs from the queue.
        """
        while True:
            try:
                # Get job from queue
                job: JobRequest = await self.queue.get()
                
                try:
                    # Run the job
                    # Note: The job functions themselves (process_image, process_pdf)
                    # handle the thread pool execution internally.
                    # We just await them here to ensure serial execution.
                    result = await job.func(*job.args, **job.kwargs)
                    
                    # Set result on future
                    if not job.future.done():
                        job.future.set_result(result)
                        
                except Exception as e:
                    self.logger.error(
                        "Error processing job",
                        component=COMPONENT_OCR_SERVICE,
                        exc_info=e
                    )
                    if not job.future.done():
                        job.future.set_exception(e)
                        
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Unexpected error in worker loop",
                    component=COMPONENT_OCR_SERVICE,
                    exc_info=e
                )
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def _enqueue_job(self, func: Callable, *args, **kwargs) -> Any:
        """
        Enqueue a job and wait for the result.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the function call
            
        Raises:
            OCRProcessingError: If queue is not initialized or full
        """
        if self.queue is None:
            # Fallback for testing or if start() wasn't called
            self.logger.warning(
                "Queue not initialized, running job directly",
                component=COMPONENT_OCR_SERVICE
            )
            return await func(*args, **kwargs)
            
        # Create future to track result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        job = JobRequest(future=future, func=func, args=args, kwargs=kwargs)
        
        try:
            # Try to put in queue with timeout (backpressure)
            # asyncio.Queue.put waits if full
            await self.queue.put(job)
            
            # Wait for result
            return await future
            
        except Exception as e:
            self.logger.error(
                "Failed to enqueue or process job",
                component=COMPONENT_OCR_SERVICE,
                exc_info=e
            )
            raise

    def load_model(self) -> None:
        """
        Load the DeepSeek-OCR model and tokenizer.
        
        Raises:
            OCRProcessingError: If model loading fails
        """
        self.model_loader.load_model()
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self.model_loader.is_loaded():
            raise OCRProcessingError(ERROR_MODEL_NOT_LOADED)
    
    async def process_image(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[str, float]:
        """
        Process an image file and extract text (Queued).
        """
        return await self._enqueue_job(
            self._process_image_impl,
            file_content,
            filename,
            prompt,
            strip_grounding
        )

    async def _process_image_impl(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[str, float]:
        """Internal implementation of process_image."""
        self._ensure_model_loaded()
        
        return await self.image_processor.process_image(
            self.model_loader.get_model(),
            self.model_loader.get_tokenizer(),
            file_content,
            filename,
            prompt,
            strip_grounding
        )
    
    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """
        Process a PDF file page by page and extract text (Queued).
        """
        return await self._enqueue_job(
            self._process_pdf_impl,
            file_content,
            filename,
            prompt,
            strip_grounding
        )

    async def _process_pdf_impl(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        strip_grounding: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """Internal implementation of process_pdf."""
        self._ensure_model_loaded()
        
        return await self.pdf_processor.process_pdf(
            self.model_loader.get_model(),
            self.model_loader.get_tokenizer(),
            file_content,
            filename,
            prompt,
            strip_grounding
        )
    
    def shutdown(self) -> None:
        """
        Shutdown the OCR service and clean up resources.
        
        This should be called when the service is no longer needed
        to ensure proper cleanup of thread pool resources.
        """
        if self._executor:
            self.logger.info(
                "Shutting down OCR service thread pool",
                component=COMPONENT_OCR_SERVICE
            )
            self._executor.shutdown(wait=True)


def get_ocr_service(config: Optional[Config] = None) -> OCRService:
    """
    Create and return an OCR service instance.
    
    Note: For dependency injection, service is stored in app state.
    This function is used for initialization.
    
    Args:
        config: Configuration object (uses default if not provided)
        
    Returns:
        OCRService: OCR service instance
    """
    return OCRService(config=config)
