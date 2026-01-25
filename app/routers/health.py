"""
Health check endpoints
"""
from fastapi import APIRouter
from ..models import HealthResponse
from ..database import test_connection
from ..config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status, database connection status, and API version.
    """
    settings = get_settings()
    db_connected = test_connection()
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        database=db_connected,
        version=settings.api_version
    )


@router.get("/")
async def root():
    """Root endpoint - redirects to docs"""
    return {
        "message": "Forecast API",
        "docs": "/docs",
        "health": "/health"
    }
