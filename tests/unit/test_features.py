"""
Unit Tests for Feature Engineering Module
==========================================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_engineering import FeatureEngineer, generate_synthetic_training_data
from src.config.settings import get_aqi_category


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return generate_synthetic_training_data(n_samples=100)
    
    def test_create_time_features(self, feature_engineer):
        """Test time-based feature creation."""
        timestamp = datetime(2024, 6, 15, 14, 30)
        features = feature_engineer._create_time_features(timestamp)
        
        assert features["hour"] == 14
        assert features["day"] == 15
        assert features["month"] == 6
        assert features["is_weekend"] == 1  # Saturday
        assert features["is_rush_hour"] == 0  # 14:30 is not rush hour
        assert "hour_sin" in features
        assert "hour_cos" in features
    
    def test_add_lag_features(self, feature_engineer, sample_df):
        """Test lag feature creation."""
        result = feature_engineer.add_lag_features(sample_df, columns=["aqi"], lags=[1, 3])
        
        assert "aqi_lag_1" in result.columns
        assert "aqi_lag_3" in result.columns
        assert len(result) == len(sample_df)
    
    def test_add_rolling_features(self, feature_engineer, sample_df):
        """Test rolling statistics feature creation."""
        result = feature_engineer.add_rolling_features(sample_df, columns=["aqi"], windows=[3, 6])
        
        assert "aqi_rolling_mean_3" in result.columns
        assert "aqi_rolling_std_3" in result.columns
        assert "aqi_rolling_mean_6" in result.columns
    
    def test_add_derived_features(self, feature_engineer, sample_df):
        """Test derived feature creation."""
        result = feature_engineer.add_derived_features(sample_df)
        
        if "aqi" in sample_df.columns:
            assert "aqi_change" in result.columns
        if "temperature" in sample_df.columns and "humidity" in sample_df.columns:
            assert "temp_humidity_product" in result.columns


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = generate_synthetic_training_data(n_samples=50)
        
        assert len(df) == 50
        assert "aqi" in df.columns
        assert "timestamp" in df.columns
        assert "pollutant_pm25" in df.columns
    
    def test_aqi_values_in_range(self):
        """Test that generated AQI values are realistic."""
        df = generate_synthetic_training_data(n_samples=100)
        
        assert df["aqi"].min() >= 0
        assert df["aqi"].max() <= 500


class TestAQICategory:
    """Tests for AQI category functions."""
    
    def test_good_category(self):
        """Test Good AQI category."""
        result = get_aqi_category(25)
        assert result["category"] == "Good"
    
    def test_moderate_category(self):
        """Test Moderate AQI category."""
        result = get_aqi_category(75)
        assert result["category"] == "Moderate"
    
    def test_unhealthy_category(self):
        """Test Unhealthy AQI category."""
        result = get_aqi_category(175)
        assert result["category"] == "Unhealthy"
    
    def test_hazardous_category(self):
        """Test Hazardous AQI category."""
        result = get_aqi_category(450)
        assert result["category"] == "Hazardous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
