"""
Application configuration using Pydantic Settings
"""
import os
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database - support both DATABASE_URL and DATABASE_PUBLIC_URL
    database_url: str = os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL", "")
    
    # API
    api_title: str = "Forecast API"
    api_version: str = "1.0.0"
    api_description: str = "Production forecast API for inventory planning"
    
    # CORS
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()
