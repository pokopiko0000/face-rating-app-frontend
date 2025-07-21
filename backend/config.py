"""
Configuration management for the Face Rating App backend.

This module provides centralized configuration management using pydantic-settings
for type safety and validation. All environment variables and application settings
are defined here.
"""

from enum import Enum
from typing import List, Tuple, Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings
import os


class Environment(str, Enum):
    """Environment types for deployment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # --- Environment Configuration ---
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    
    # --- Server Configuration ---
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    
    # --- API Configuration ---
    api_title: str = Field(default="Face Rating API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # --- External Services ---
    r2_public_url: str = Field(env="R2_PUBLIC_URL")
    gcs_bucket_name: str = Field(env="GCS_BUCKET_NAME")
    
    # --- CORS Configuration ---
    cors_origins: List[str] = Field(default_factory=list, env="CORS_ORIGINS")
    
    # --- Face Analysis Settings ---
    face_detection_threshold: float = Field(default=0.1, env="FACE_DETECTION_THRESHOLD")
    face_detection_size: Tuple[int, int] = Field(default=(640, 640), env="FACE_DETECTION_SIZE")
    face_providers: List[str] = Field(default=["CPUExecutionProvider"], env="FACE_PROVIDERS")
    
    # --- Scoring Configuration ---
    geo_bonus: float = Field(default=0.05, env="GEO_BONUS")
    rarity_bonus_unit: float = Field(default=0.01, env="RARITY_BONUS_UNIT")
    
    # --- Debug and Logging ---
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @validator("face_detection_size", pre=True)
    def parse_detection_size(cls, v):
        """Parse face detection size tuple."""
        if isinstance(v, str):
            width, height = map(int, v.split(","))
            return (width, height)
        return v
    
    @validator("face_providers", pre=True)
    def parse_face_providers(cls, v):
        """Parse face analysis providers."""
        if isinstance(v, str):
            return [provider.strip() for provider in v.split(",") if provider.strip()]
        return v
    
    @validator("r2_public_url")
    def validate_r2_url(cls, v):
        """Validate R2 URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("R2 URL must be a valid HTTP/HTTPS URL")
        return v
    
    @validator("geo_bonus", "rarity_bonus_unit", "face_detection_threshold")
    def validate_positive_float(cls, v):
        """Validate positive float values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration based on environment."""
        if self.environment == Environment.PRODUCTION:
            if not self.cors_origins:
                raise ValueError("CORS origins must be specified in production")
            return {
                "allow_origins": self.cors_origins,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["*"],
            }
        elif self.environment == Environment.DEVELOPMENT:
            # For development, allow specific origins or localhost
            default_dev_origins = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ]
            origins = self.cors_origins if self.cors_origins else default_dev_origins
            return {
                "allow_origins": origins,
                "allow_credentials": False,  # Disable credentials for development
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }
        else:  # Testing
            return {
                "allow_origins": ["*"],
                "allow_credentials": False,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }
    
    def get_face_analysis_config(self) -> dict:
        """Get face analysis model configuration."""
        return {
            "providers": self.face_providers,
            "det_thresh": self.face_detection_threshold,
            "det_size": self.face_detection_size,
        }


# Global settings instance
settings = Settings()

# Convenience functions for backward compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings