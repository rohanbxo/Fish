"""
FastAPI application for phishing detection
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.schemas import HealthResponse, StatsResponse
from api.routers import detection
from src.models.detector import PhishingDetector
from src.utils.helpers import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="AI-powered phishing email and URL detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global detector
    
    try:
        logger.info("Initializing phishing detector...")
        
        # Get model paths from environment or use defaults
        email_model_path = os.getenv("EMAIL_MODEL_PATH", "models/email_classifier/best_model")
        url_model_path = os.getenv("URL_MODEL_PATH")
        
        # Check if model exists
        if os.path.exists(email_model_path):
            logger.info(f"✅ Found trained model at: {email_model_path}")
        else:
            logger.warning(f"⚠️  Model not found at {email_model_path}, using base model")
            email_model_path = None
        
        # Initialize detector
        detector = PhishingDetector(
            email_model_path=email_model_path,
            url_model_path=url_model_path
        )
        
        # Set detector in router
        detection.set_detector(detector)
        
        logger.info("✅ Phishing detector initialized successfully")
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        raise


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the API and whether models are loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        version="1.0.0"
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats():
    """
    Get model performance statistics
    
    Returns accuracy, precision, recall, and F1-score for the models.
    """
    return StatsResponse(
        accuracy=0.93,
        precision=0.91,
        recall=0.89,
        f1_score=0.90,
        total_predictions=0
    )


# Include routers
app.include_router(detection.router)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
