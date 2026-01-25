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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    
    # CORS middleware - allow all origins explicitly
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Must be False when allow_origins=["*"]
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Global exception handler with CORS headers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
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
