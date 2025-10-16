"""
API schemas using Pydantic
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, EmailStr, Field


class EmailRequest(BaseModel):
    """Request model for email detection"""
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    sender: EmailStr = Field(..., description="Sender email address")
    urls: Optional[List[str]] = Field(default=[], description="List of URLs in email")
    
    class Config:
        schema_extra = {
            "example": {
                "subject": "Urgent: Verify Your Account",
                "body": "Dear user, your account will be suspended unless you verify your information at http://suspicious-site.com",
                "sender": "noreply@fake-bank.com",
                "urls": ["http://suspicious-site.com"]
            }
        }


class DetectionResponse(BaseModel):
    """Response model for detection results"""
    risk_score: float = Field(..., description="Overall risk score (0-1)")
    risk_level: str = Field(..., description="Risk level category")
    email_score: float = Field(..., description="Email text risk score (0-1)")
    url_scores: List[float] = Field(default=[], description="Individual URL risk scores")
    urls: List[str] = Field(default=[], description="URLs analyzed")
    explanation: Dict = Field(..., description="Detailed explanation of results")
    indicators: List[str] = Field(default=[], description="Detected phishing indicators")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_score": 0.85,
                "risk_level": "HIGH - Likely Phishing",
                "email_score": 0.82,
                "url_scores": [0.91],
                "urls": ["http://suspicious-site.com"],
                "explanation": {
                    "summary": "High confidence phishing detection",
                    "email_analysis": {
                        "phishing_probability": "82.0%",
                        "urgency_indicators": 2,
                        "sensitive_info_requests": 1
                    },
                    "url_analysis": {
                        "total_urls": 1,
                        "max_risk_score": "91.0%",
                        "high_risk_urls": 1
                    }
                },
                "indicators": [
                    "Contains 2 urgency keyword(s)",
                    "Requests sensitive information",
                    "1 suspicious URL(s) detected"
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """Model statistics response"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int = 0
