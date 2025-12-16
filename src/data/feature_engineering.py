"""
Feature Engineering Module
==========================

This module transforms raw AQI and weather data into ML-ready features.

Feature Categories:
1. Raw Pollutant Values - PM2.5, PM10, O3, NO2, SO2, CO
2. Weather Features - Temperature, Humidity, Wind, Pressure
3. Time-Based Features - Hour, Day, Month, DayOfWeek, IsWeekend
4. Lag Features - Previous hour/day values
5. Rolling Statistics - 3h, 6h, 12h, 24h averages and std devs
6. Derived Features - AQI change rate, pollution ratios

Why Feature Engineering Matters:
- ML models work best with meaningful features
- Time-based features capture daily/seasonal patterns
- Lag features help with time series prediction
- Rolling statistics smooth out noise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from src.config.settings import get_settings, get_aqi_category
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for AQI prediction.
    
    This class transforms raw data into features suitable for ML models.
    It handles both single observations and batch processing.
    """
    
    # AQI category encoding for classification
    AQI_CATEGORY_ENCODING = {
        "Good": 0,
        "Moderate": 1,
        "Unhealthy for Sensitive Groups": 2,
        "Unhealthy": 3,
        "Very Unhealthy": 4,
        "Hazardous": 5
    }
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.settings = get_settings()
        self.features_dir = Path(self.settings.app.features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
    
    @log_execution_time
    def create_features_from_raw(
        self,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create features from a single raw data observation.
        
        Args:
            raw_data: Raw data from DataIngestionService.fetch_all_data()
        
        Returns:
            Dictionary of engineered features
        
        Example:
            >>> fe = FeatureEngineer()
            >>> raw_data = ingestion_service.fetch_all_data("beijing")
            >>> features = fe.create_features_from_raw(raw_data)
        """
        features = {}
        
        # Extract timestamp
        timestamp_str = raw_data.get("fetch_timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
            features.update(self._create_time_features(timestamp))
        
        # Extract AQI features
        aqi_data = raw_data.get("aqi_data", {})
        if aqi_data and "error" not in aqi_data:
            features.update(self._extract_aqi_features(aqi_data))
        
        # Extract weather features
        weather_data = raw_data.get("current_weather", {})
        if weather_data and "error" not in weather_data:
            features.update(self._extract_weather_features(weather_data))
        
        # Add metadata
        features["city"] = raw_data.get("city", "unknown")
        features["timestamp"] = timestamp_str
        
        logger.debug(f"Created {len(features)} features from raw data")
        
        return features
    
    def _create_time_features(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Create time-based features from timestamp.
        
        Time features help capture:
        - Daily patterns (rush hour pollution)
        - Weekly patterns (weekday vs weekend)
        - Seasonal patterns (winter heating, summer ozone)
        """
        return {
            # Basic time components
            "hour": timestamp.hour,
            "day": timestamp.day,
            "month": timestamp.month,
            "year": timestamp.year,
            "day_of_week": timestamp.weekday(),  # 0=Monday, 6=Sunday
            "day_of_year": timestamp.timetuple().tm_yday,
            "week_of_year": timestamp.isocalendar()[1],
            
            # Binary flags
            "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
            "is_rush_hour": 1 if timestamp.hour in [7, 8, 9, 17, 18, 19] else 0,
            "is_night": 1 if timestamp.hour < 6 or timestamp.hour >= 22 else 0,
            
            # Cyclical encoding (useful for ML models)
            # Transforms hour/month into sin/cos to capture cyclical nature
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
            "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
            "month_cos": np.cos(2 * np.pi * timestamp.month / 12),
            "day_of_week_sin": np.sin(2 * np.pi * timestamp.weekday() / 7),
            "day_of_week_cos": np.cos(2 * np.pi * timestamp.weekday() / 7),
        }
    
    def _extract_aqi_features(self, aqi_data: Dict) -> Dict[str, Any]:
        """
        Extract AQI-related features.
        
        Includes:
        - Raw AQI value (target for regression)
        - AQI category (target for classification)
        - Individual pollutant concentrations
        """
        features = {
            # Target variables
            "aqi": aqi_data.get("aqi", 0),
            "aqi_category": self._encode_category(aqi_data.get("category", "Unknown")),
            "aqi_category_name": aqi_data.get("category", "Unknown"),
            
            # Dominant pollutant (one-hot encode later)
            "dominant_pollutant": aqi_data.get("dominant_pollutant", "unknown"),
        }
        
        # Pollutant values
        pollutants = aqi_data.get("pollutants", {})
        for pollutant in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            value = pollutants.get(pollutant)
            features[f"pollutant_{pollutant}"] = value if value is not None else np.nan
        
        # Weather from AQICN (may differ from OpenWeather)
        aqicn_weather = aqi_data.get("weather", {})
        features["aqicn_temperature"] = aqicn_weather.get("temperature")
        features["aqicn_humidity"] = aqicn_weather.get("humidity")
        features["aqicn_pressure"] = aqicn_weather.get("pressure")
        features["aqicn_wind"] = aqicn_weather.get("wind")
        
        return features
    
    def _extract_weather_features(self, weather_data: Dict) -> Dict[str, Any]:
        """
        Extract weather-related features.
        
        Weather strongly influences air quality:
        - Wind disperses pollutants
        - Rain washes out particles
        - Temperature affects chemical reactions
        - Humidity affects particle formation
        """
        return {
            "temperature": weather_data.get("temperature"),
            "feels_like": weather_data.get("feels_like"),
            "temp_min": weather_data.get("temp_min"),
            "temp_max": weather_data.get("temp_max"),
            "humidity": weather_data.get("humidity"),
            "pressure": weather_data.get("pressure"),
            "wind_speed": weather_data.get("wind_speed"),
            "wind_direction": weather_data.get("wind_direction"),
            "visibility": weather_data.get("visibility"),
            "clouds": weather_data.get("clouds"),
            
            # Weather condition (categorical)
            "weather_condition": weather_data.get("weather_condition", "Unknown"),
        }
    
    def _encode_category(self, category: str) -> int:
        """Encode AQI category to numeric value."""
        return self.AQI_CATEGORY_ENCODING.get(category, -1)
    
    @log_execution_time
    def create_features_dataframe(
        self,
        raw_data_list: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a DataFrame of features from multiple raw observations.
        
        Args:
            raw_data_list: List of raw data dictionaries
        
        Returns:
            DataFrame with all features
        """
        features_list = []
        
        for raw_data in raw_data_list:
            try:
                features = self.create_features_from_raw(raw_data)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to create features: {e}")
                continue
        
        if not features_list:
            logger.warning("No features created from raw data")
            return pd.DataFrame()
        
        df = pd.DataFrame(features_list)
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    @log_execution_time
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        lags: List[int] = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add lag features (previous values) for specified columns.
        
        Lag features are crucial for time series prediction:
        - lag_1: Previous observation (1 hour ago)
        - lag_24: Same time yesterday
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for (default: AQI and pollutants)
            lags: Lag periods (in hours, assuming hourly data)
        
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        if columns is None:
            columns = ["aqi", "pollutant_pm25", "pollutant_pm10"]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"
                df[lag_col_name] = df[col].shift(lag)
                logger.debug(f"Created lag feature: {lag_col_name}")
        
        return df
    
    @log_execution_time
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        windows: List[int] = [3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add rolling statistics (moving averages, std devs) for specified columns.
        
        Rolling features smooth out noise and capture trends:
        - rolling_mean_24: Average over past 24 hours
        - rolling_std_24: Volatility over past 24 hours
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate rolling stats for
            windows: Window sizes (in hours)
        
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        if columns is None:
            columns = ["aqi", "pollutant_pm25", "temperature", "humidity"]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                mean_col = f"{col}_rolling_mean_{window}"
                df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                std_col = f"{col}_rolling_std_{window}"
                df[std_col] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max for AQI
                if col == "aqi":
                    df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window, min_periods=1).min()
                    df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window, min_periods=1).max()
        
        return df
    
    @log_execution_time
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features calculated from existing features.
        
        These are domain-specific features that might have predictive power:
        - AQI change rate
        - Pollution ratios
        - Temperature-humidity interaction
        """
        df = df.copy()
        
        # AQI change rate (hour over hour)
        if "aqi" in df.columns:
            df["aqi_change"] = df["aqi"].diff()
            df["aqi_change_pct"] = df["aqi"].pct_change() * 100
        
        # PM2.5 to PM10 ratio (indicates particle size distribution)
        if "pollutant_pm25" in df.columns and "pollutant_pm10" in df.columns:
            df["pm25_pm10_ratio"] = df["pollutant_pm25"] / (df["pollutant_pm10"] + 0.001)
        
        # Temperature-humidity interaction (affects particle formation)
        if "temperature" in df.columns and "humidity" in df.columns:
            df["temp_humidity_product"] = df["temperature"] * df["humidity"]
            # Heat index approximation
            df["heat_index"] = -8.785 + 1.611*df["temperature"] + 2.339*df["humidity"]
        
        # Wind chill effect on pollution dispersion
        if "wind_speed" in df.columns:
            df["wind_effect"] = np.log1p(df["wind_speed"])
        
        return df
    
    def get_feature_names(self, include_targets: bool = False) -> List[str]:
        """
        Get list of feature names for model training.
        
        Args:
            include_targets: Whether to include target variables
        
        Returns:
            List of feature column names
        """
        # Core features (always included)
        features = [
            # Time features
            "hour", "day", "month", "day_of_week", "day_of_year",
            "is_weekend", "is_rush_hour", "is_night",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            
            # Pollutant features
            "pollutant_pm25", "pollutant_pm10", "pollutant_o3",
            "pollutant_no2", "pollutant_so2", "pollutant_co",
            
            # Weather features
            "temperature", "humidity", "pressure", "wind_speed",
            "visibility", "clouds",
        ]
        
        if include_targets:
            features.extend(["aqi", "aqi_category"])
        
        return features
    
    @log_execution_time
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = "aqi",
        forecast_horizon: int = 24
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Feature DataFrame
            target_column: Target variable column name
            forecast_horizon: Hours ahead to predict
        
        Returns:
            Tuple of (X features, y target)
        """
        df = df.copy()
        
        # Add lag and rolling features
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_derived_features(df)
        
        # Create target variable (future AQI)
        df[f"target_{target_column}"] = df[target_column].shift(-forecast_horizon)
        
        # Drop rows with NaN (at beginning due to lags, at end due to target shift)
        df = df.dropna()
        
        # Separate features and target
        target_col = f"target_{target_column}"
        feature_cols = [col for col in df.columns if col not in [
            target_col, "timestamp", "city", "aqi_category_name",
            "dominant_pollutant", "weather_condition"
        ]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def save_features(
        self,
        df: pd.DataFrame,
        filename: str = None
    ) -> Path:
        """
        Save features to a file.
        
        Args:
            df: Feature DataFrame
            filename: Output filename (default: features_YYYYMMDD_HHMMSS.parquet)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"features_{timestamp}.parquet"
        
        filepath = self.features_dir / filename
        
        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved features to {filepath}")
        
        return filepath
    
    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load features from a parquet file."""
        return pd.read_parquet(filepath)


def generate_synthetic_training_data(
    n_samples: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic training data for testing purposes.
    
    This creates realistic-looking AQI data with patterns:
    - Daily cycles (rush hour peaks)
    - Weekly patterns (lower on weekends)
    - Seasonal patterns (winter heating)
    - Random noise
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic features and AQI values
    """
    np.random.seed(seed)
    
    logger.info(f"Generating {n_samples} synthetic samples")
    
    # Generate timestamps (hourly for 1000 hours ≈ 41 days)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Create base patterns
    hours = np.array([t.hour for t in timestamps])
    days = np.array([t.weekday() for t in timestamps])
    months = np.array([t.month for t in timestamps])
    
    # Daily pattern: Higher during rush hours
    daily_pattern = 20 * np.sin((hours - 8) * np.pi / 12) + 30
    
    # Weekly pattern: Lower on weekends
    weekly_pattern = np.where(days >= 5, -15, 0)
    
    # Seasonal pattern: Higher in winter
    seasonal_pattern = 20 * np.cos((months - 1) * np.pi / 6)
    
    # Generate AQI with patterns + noise
    base_aqi = 80  # Base AQI level
    noise = np.random.normal(0, 15, n_samples)
    aqi = base_aqi + daily_pattern + weekly_pattern + seasonal_pattern + noise
    aqi = np.clip(aqi, 10, 400)  # Realistic AQI range
    
    # Generate correlated pollutants
    pm25 = aqi * 0.4 + np.random.normal(0, 10, n_samples)
    pm10 = pm25 * 1.5 + np.random.normal(0, 15, n_samples)
    o3 = 40 + 20 * np.sin(hours * np.pi / 12) + np.random.normal(0, 10, n_samples)
    no2 = aqi * 0.3 + np.random.normal(0, 8, n_samples)
    so2 = aqi * 0.1 + np.random.normal(0, 5, n_samples)
    co = aqi * 0.02 + np.random.normal(0, 0.5, n_samples)
    
    # Generate weather features
    temperature = 15 + 10 * np.sin((months - 7) * np.pi / 6) + np.random.normal(0, 5, n_samples)
    humidity = 60 + np.random.normal(0, 15, n_samples)
    humidity = np.clip(humidity, 20, 100)
    pressure = 1013 + np.random.normal(0, 10, n_samples)
    wind_speed = np.abs(np.random.normal(3, 2, n_samples))
    
    # Create DataFrame
    fe = FeatureEngineer()
    
    data = []
    for i in range(n_samples):
        row = {
            "timestamp": timestamps[i].isoformat(),
            "city": "synthetic_city",
            "aqi": aqi[i],
            "aqi_category": fe._encode_category(get_aqi_category(aqi[i])["category"]),
            "aqi_category_name": get_aqi_category(aqi[i])["category"],
            "pollutant_pm25": max(pm25[i], 0),
            "pollutant_pm10": max(pm10[i], 0),
            "pollutant_o3": max(o3[i], 0),
            "pollutant_no2": max(no2[i], 0),
            "pollutant_so2": max(so2[i], 0),
            "pollutant_co": max(co[i], 0),
            "temperature": temperature[i],
            "humidity": humidity[i],
            "pressure": pressure[i],
            "wind_speed": wind_speed[i],
            "visibility": 10000 - aqi[i] * 15,  # Lower visibility with higher AQI
            "clouds": np.random.randint(0, 100),
        }
        
        # Add time features
        row.update(fe._create_time_features(timestamps[i]))
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info(f"Generated synthetic data with {len(df)} rows")
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering Module")
    print("=" * 50)
    
    # Generate synthetic data
    print("\n1. Generating synthetic training data:")
    df = generate_synthetic_training_data(n_samples=500)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:10]}...")
    print(f"   AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    
    # Add lag features
    print("\n2. Adding lag features:")
    fe = FeatureEngineer()
    df_with_lags = fe.add_lag_features(df)
    lag_cols = [c for c in df_with_lags.columns if "lag" in c]
    print(f"   Added {len(lag_cols)} lag features")
    
    # Add rolling features
    print("\n3. Adding rolling features:")
    df_with_rolling = fe.add_rolling_features(df_with_lags)
    rolling_cols = [c for c in df_with_rolling.columns if "rolling" in c]
    print(f"   Added {len(rolling_cols)} rolling features")
    
    # Add derived features
    print("\n4. Adding derived features:")
    df_final = fe.add_derived_features(df_with_rolling)
    print(f"   Final shape: {df_final.shape}")
    
    # Prepare training data
    print("\n5. Preparing training data:")
    X, y = fe.prepare_training_data(df, target_column="aqi", forecast_horizon=24)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Save features
    print("\n6. Saving features:")
    filepath = fe.save_features(df_final, "test_features.parquet")
    print(f"   Saved to: {filepath}")
    
    print("\n" + "=" * 50)
    print("Feature Engineering Test Complete!")
