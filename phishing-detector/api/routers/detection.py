"""
Detection router for API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from ..schemas import EmailRequest, DetectionResponse
from src.models.detector import PhishingDetector

router = APIRouter(prefix="/api/v1", tags=["detection"])
logger = logging.getLogger(__name__)

# Initialize detector (will be set by main app)
detector: PhishingDetector = None


def set_detector(phishing_detector: PhishingDetector):
    """Set the detector instance"""
    global detector
    detector = phishing_detector


@router.post("/detect", response_model=DetectionResponse)
async def detect_phishing(request: EmailRequest):
    """
    Analyze email and URLs for phishing indicators
    
    - **subject**: Email subject line
    - **body**: Email body content
    - **sender**: Sender email address
    - **urls**: Optional list of URLs to analyze
    
    Returns detection results with risk score and detailed analysis.
    """
    try:
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Detection service not available"
            )
        
        # Run detection
        result = detector.detect(
            subject=request.subject,
            body=request.body,
            sender=request.sender,
            urls=request.urls if request.urls else None
        )
        
        logger.info(
            f"Detection completed - Risk: {result['risk_level']}, "
            f"Score: {result['risk_score']:.2f}"
        )
        
        return DetectionResponse(**result)
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch", response_model=List[DetectionResponse])
async def detect_phishing_batch(requests: List[EmailRequest]):
    """
    Analyze multiple emails for phishing indicators
    
    Accepts a list of email requests and returns detection results for each.
    """
    try:
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Detection service not available"
            )
        
        # Prepare emails for batch detection
        emails = [
            {
                'subject': req.subject,
                'body': req.body,
                'sender': req.sender,
                'urls': req.urls if req.urls else None
            }
            for req in requests
        ]
        
        # Run batch detection
        results = detector.batch_detect(emails)
        
        logger.info(f"Batch detection completed for {len(results)} emails")
        
        return [DetectionResponse(**result) for result in results]
    
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
