"""
Configuration management for the Face Rating App Backend.

This module handles environment variables and application settings using pydantic-settings.
"""

import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environment enumeration for application deployment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """Application configuration settings."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8003
    
    # External services
    r2_public_url: str = "https://pub-20801d1056e542a99ab766366e3a3124.r2.dev"
    gcs_bucket_name: str = "imagen4-faces-imagen-demo-460715"
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Face analysis settings
    geo_bonus: float = 0.05  # Bonus for different continents (5%)
    rarity_bonus_unit: float = 0.01  # Bonus per rarity star (1%)
    det_thresh: float = 0.1  # Face detection threshold
    det_size: tuple = (640, 640)  # Detection size
    
    # Model settings
    face_analysis_providers: List[str] = ["CPUExecutionProvider"]
    confidence_threshold: float = 0.25
    min_face_size: tuple = (30, 30)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "FACE_RATING_"
    
    @property
    def cors_config(self) -> dict:
        """Get CORS configuration based on environment."""
        if self.environment == Environment.PRODUCTION:
            return {
                "allow_origins": self.cors_origins,
                "allow_credentials": self.cors_allow_credentials,
                "allow_methods": self.cors_allow_methods,
                "allow_headers": self.cors_allow_headers,
            }
        else:
            # Development mode - allow all origins
            return {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT


# Global settings instance
settings = AppSettings()