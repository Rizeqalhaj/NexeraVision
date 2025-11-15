"""
NexaraVision ML Service - FastAPI Application

Production-ready violence detection API with GPU acceleration.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import detect, detect_live, websocket
from app.core.config import settings
from app.core.gpu import configure_gpu, get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Configure GPU
    gpu_available = configure_gpu(settings.GPU_MEMORY_FRACTION)

    if gpu_available:
        logger.info("GPU configuration successful")
    else:
        logger.warning("Running on CPU - inference will be slower")

    # Initialize model
    logger.info("Loading violence detection model...")
    detect.initialize_detector()
    detect_live.detector = detect.detector  # Share model instance with HTTP endpoint
    websocket.detector = detect.detector  # Share model instance with WebSocket endpoint
    logger.info("Model loaded and ready")

    # Log device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")

    yield

    # Shutdown
    logger.info("Shutting down ML service")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Violence detection ML service with real-time and batch processing",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect.router, prefix="/api", tags=["detection"])
app.include_router(detect_live.router, prefix="/api", tags=["live-detection"])
app.include_router(websocket.router, prefix="/api", tags=["websocket"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "endpoints": {
            "file_upload": "/api/detect",
            "live_stream": "/api/detect_live",
            "websocket_live": "/api/ws/live",
            "websocket_status": "/api/ws/status",
            "batch_processing": "/api/detect_live_batch",
            "health": "/api/health",
            "docs": "/docs",
        }
    }


@app.get("/api/info")
async def service_info():
    """Get service and device information."""
    device_info = get_device_info()

    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_path": settings.MODEL_PATH,
        "configuration": {
            "num_frames": settings.NUM_FRAMES,
            "frame_size": settings.FRAME_SIZE,
            "batch_size": settings.BATCH_SIZE,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE / (1024 * 1024),
        },
        "device": device_info,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
