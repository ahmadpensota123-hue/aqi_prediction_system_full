"""
Application Settings Configuration
==================================

This module handles all configuration settings for the AQI Prediction System.
It uses Pydantic Settings to manage environment variables and provide
type-safe configuration access throughout the application.

Why Pydantic Settings?
- Type validation for all config values
- Automatic .env file loading
- Easy to test with overrides
- Industry standard for FastAPI projects
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """
    API Configuration Settings
    
    These settings control external API connections for data sources.
    You need to obtain free API keys from:
    - AQICN: https://aqicn.org/data-platform/token/
    - OpenWeather: https://openweathermap.org/api
    """
    
    # AQICN API for air quality data
    aqicn_api_key: str = "demo"  # "demo" works for testing but has limits
    aqicn_base_url: str = "https://api.waqi.info"
    
    # OpenWeather API for weather data
    openweather_api_key: str = ""
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class FeatureStoreSettings(BaseSettings):
    """
    Hopsworks Feature Store Settings
    
    Hopsworks provides a free tier for feature storage and versioning.
    Sign up at: https://app.hopsworks.ai/
    """
    
    hopsworks_api_key: str = ""
    hopsworks_project_name: str = "aqi_prediction"
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class MLflowSettings(BaseSettings):
    """
    MLflow Experiment Tracking Settings
    
    MLflow is used to track experiments, log metrics, and store models.
    """
    
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "aqi_prediction"
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class PrefectSettings(BaseSettings):
    """
    Prefect Workflow Orchestration Settings
    """
    
    prefect_api_url: str = "http://localhost:4200/api"
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class LocationSettings(BaseSettings):
    """
    Default Location Settings for AQI Predictions
    
    These define the default city/location for predictions.
    Can be overridden per request via the API.
    """
    
    default_city: str = "beijing"
    default_latitude: float = 39.9042
    default_longitude: float = 116.4074
    
    model_config = SettingsConfigDict(
        env_prefix="DEFAULT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class AppSettings(BaseSettings):
    """
    Main Application Settings
    
    Central configuration class that combines all settings.
    Loads values from environment variables and .env file.
    """
    
    # Application metadata
    app_name: str = "AQI Prediction System"
    app_version: str = "1.0.0"
    app_env: str = "development"  # development, staging, production
    
    # Logging
    log_level: str = "INFO"
    debug: bool = True
    
    # API server settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Dashboard settings
    dashboard_port: int = 8501
    
    # Data directories
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    features_dir: str = "data/features"
    models_dir: str = "models"
    
    # Model settings
    forecast_days: int = 3
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class Settings:
    """
    Combined Settings Manager
    
    This class provides a unified interface to access all settings.
    It's the main entry point for configuration throughout the app.
    
    Example Usage:
        from src.config.settings import get_settings
        
        settings = get_settings()
        print(settings.app.app_name)
        print(settings.api.aqicn_api_key)
    """
    
    def __init__(self):
        self.app = AppSettings()
        self.api = APISettings()
        self.feature_store = FeatureStoreSettings()
        self.mlflow = MLflowSettings()
        self.prefect = PrefectSettings()
        self.location = LocationSettings()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are only loaded once
    and reused throughout the application lifecycle.
    
    Returns:
        Settings: The application settings instance
    """
    return Settings()


# AQI Categories based on US EPA standards
AQI_CATEGORIES = {
    "Good": {
        "min": 0, "max": 50, 
        "color": "#00e400", 
        "health_message": "Air quality is satisfactory",
        "precautions": [
            "Enjoy outdoor activities.",
            "Ventilate your home."
        ]
    },
    "Moderate": {
        "min": 51, "max": 100, 
        "color": "#ffff00", 
        "health_message": "Acceptable; moderate health concern for sensitive groups",
        "precautions": [
            "Sensitive groups should reduce outdoor exercise.",
            "Wear a mask if you have respiratory issues."
        ]
    },
    "Unhealthy for Sensitive Groups": {
        "min": 101, "max": 150, 
        "color": "#ff7e00", 
        "health_message": "Sensitive groups may experience health effects",
        "precautions": [
            "Children & elderly should reduce outdoor exertion.",
            "Wear a mask if sensitive.",
            "Close windows to avoid outdoor air."
        ]
    },
    "Unhealthy": {
        "min": 151, "max": 200, 
        "color": "#ff0000", 
        "health_message": "Everyone may begin to experience health effects",
        "precautions": [
            "Wear a mask outdoors (N95 recommended).",
            "Avoid prolonged outdoor exertion.",
            "Run an air purifier indoors."
        ]
    },
    "Very Unhealthy": {
        "min": 201, "max": 300, 
        "color": "#8f3f97", 
        "health_message": "Health alert: everyone may experience serious effects",
        "precautions": [
            "Avoid all outdoor activities.",
            "Wear an N95 mask if you must go out.",
            "Keep windows and doors closed.",
            "Use air purifiers on high."
        ]
    },
    "Hazardous": {
        "min": 301, "max": 500, 
        "color": "#7e0023", 
        "health_message": "Health emergency: everyone is likely to be affected",
        "precautions": [
            "Stay indoors strictly.",
            "Seal windows and doors.",
            "Wear an N95/P100 mask immediately if outside.",
            "Seek medical attention if you feel breathless."
        ]
    }
}


def get_aqi_category(aqi_value: float) -> dict:
    """
    Get AQI category information based on AQI value.
    
    Args:
        aqi_value: The AQI value (0-500+)
    
    Returns:
        dict: Category information including name, color, and health message
    
    Example:
        >>> get_aqi_category(75)
        {'category': 'Moderate', 'color': '#ffff00', 'health_message': '...'}
    """
    for category, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi_value <= info["max"]:
            return {
                "category": category,
                "color": info["color"],
                "health_message": info["health_message"],
                "precautions": info.get("precautions", [])
            }
    
    # Handle values above 500 (Hazardous)
    return {
        "category": "Hazardous",
        "color": "#7e0023",
        "health_message": "Health emergency: everyone is likely to be affected",
        "precautions": AQI_CATEGORIES["Hazardous"]["precautions"]
    }


if __name__ == "__main__":
    # Test settings loading
    settings = get_settings()
    print(f"App Name: {settings.app.app_name}")
    print(f"Environment: {settings.app.app_env}")
    print(f"Log Level: {settings.app.log_level}")
    print(f"Default City: {settings.location.default_city}")
    print(f"AQICN API Key: {'Set' if settings.api.aqicn_api_key else 'Not Set'}")
