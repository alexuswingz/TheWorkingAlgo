"""
Forecast API - FastAPI Application
================================================================================
Production-ready API for inventory forecast calculations.

Endpoints:
- GET /health - Health check
- GET /forecast/{asin} - Get forecast for a single ASIN
- GET /forecast/ - Get pre-aggregated forecasts for all products

Algorithms (LOCKED - DO NOT MODIFY):
- 0-6m: Peak-based forecasting with seasonality elasticity
- 6-18m: CVR-based forecasting with seasonality weighting
- 18m+: Weighted smoothing with prior year data
================================================================================
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import get_settings
from .routers import forecast, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.api_title} v{settings.api_version}")
    yield
    # Shutdown
    print("Shutting down...")


def create_app() -> FastAPI:
    """Application factory"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(forecast.router)
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
