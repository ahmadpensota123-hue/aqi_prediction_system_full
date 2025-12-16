"""
Hopsworks Feature Store Integration
===================================

This module provides integration with Hopsworks, a feature store platform.

What is a Feature Store?
- Centralized repository for ML features
- Enables feature sharing across models and teams
- Handles versioning and lineage tracking
- Provides point-in-time correct feature retrieval
- Reduces training-serving skew

Why Hopsworks?
- Free tier available (perfect for educational projects)
- Python-first API
- Supports online and offline feature stores
- Great documentation and community

Key Concepts:
- Feature Group: Collection of related features (like a table)
- Feature View: Interface for reading features for training/inference
- Training Dataset: Materialized features for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class HopsworksClient:
    """
    Client for interacting with Hopsworks Feature Store.
    
    This client handles:
    - Connection management
    - Feature group creation
    - Feature ingestion
    - Feature retrieval
    
    Usage:
        >>> client = HopsworksClient()
        >>> client.connect()
        >>> client.create_feature_group(df, "aqi_features")
        >>> features = client.get_features(["aqi", "pm25"])
    """
    
    def __init__(self, api_key: str = None, project_name: str = None):
        """
        Initialize Hopsworks client.
        
        Args:
            api_key: Hopsworks API key (or from environment)
            project_name: Hopsworks project name
        """
        settings = get_settings()
        self.api_key = api_key or settings.feature_store.hopsworks_api_key
        self.project_name = project_name or settings.feature_store.hopsworks_project_name
        
        self._connection = None
        self._project = None
        self._fs = None
        
        if not self.api_key:
            logger.warning(
                "Hopsworks API key not set. "
                "Get one at: https://app.hopsworks.ai/"
            )
    
    def connect(self) -> bool:
        """
        Connect to Hopsworks.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot connect: Hopsworks API key not set")
            return False
        
        try:
            import hopsworks
            
            logger.info(f"Connecting to Hopsworks project: {self.project_name}")
            
            self._connection = hopsworks.login(api_key_value=self.api_key)
            self._project = self._connection.get_project(self.project_name)
            self._fs = self._project.get_feature_store()
            
            logger.info("Successfully connected to Hopsworks")
            return True
            
        except ImportError:
            logger.error("hopsworks package not installed. Run: pip install hopsworks")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Hopsworks."""
        return self._fs is not None
    
    @log_execution_time
    def create_feature_group(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = None,
        primary_key: List[str] = None,
        event_time: str = None,
        version: int = 1
    ) -> Any:
        """
        Create or update a feature group.
        
        A feature group is like a table of features that can be
        used for training and inference.
        
        Args:
            df: DataFrame with features
            name: Feature group name
            description: Description of the feature group
            primary_key: Primary key columns
            event_time: Timestamp column for point-in-time queries
            version: Feature group version
        
        Returns:
            Created feature group object
        """
        if not self.is_connected():
            logger.error("Not connected to Hopsworks")
            return None
        
        if primary_key is None:
            primary_key = ["city", "timestamp"] if "timestamp" in df.columns else ["city"]
        
        if event_time is None and "timestamp" in df.columns:
            event_time = "timestamp"
        
        try:
            # Get or create feature group
            fg = self._fs.get_or_create_feature_group(
                name=name,
                version=version,
                description=description or f"AQI prediction features - {name}",
                primary_key=primary_key,
                event_time=event_time,
            )
            
            # Insert data
            fg.insert(df)
            
            logger.info(f"Created/updated feature group: {name} v{version} with {len(df)} rows")
            
            return fg
            
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            return None
    
    @log_execution_time
    def get_feature_group(
        self,
        name: str,
        version: int = 1
    ) -> Any:
        """
        Get an existing feature group.
        
        Args:
            name: Feature group name
            version: Feature group version
        
        Returns:
            Feature group object or None
        """
        if not self.is_connected():
            logger.error("Not connected to Hopsworks")
            return None
        
        try:
            fg = self._fs.get_feature_group(name=name, version=version)
            logger.info(f"Retrieved feature group: {name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to get feature group: {e}")
            return None
    
    @log_execution_time
    def create_feature_view(
        self,
        name: str,
        feature_group_name: str,
        features: List[str] = None,
        version: int = 1
    ) -> Any:
        """
        Create a feature view for training/inference.
        
        A feature view defines which features to use from
        one or more feature groups.
        
        Args:
            name: Feature view name
            feature_group_name: Source feature group
            features: List of feature columns to include
            version: Feature view version
        
        Returns:
            Feature view object
        """
        if not self.is_connected():
            logger.error("Not connected to Hopsworks")
            return None
        
        try:
            fg = self.get_feature_group(feature_group_name, version)
            if fg is None:
                return None
            
            # If specific features requested, select them
            if features:
                query = fg.select(features)
            else:
                query = fg.select_all()
            
            # Create feature view
            fv = self._fs.get_or_create_feature_view(
                name=name,
                version=version,
                query=query
            )
            
            logger.info(f"Created feature view: {name} v{version}")
            
            return fv
            
        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            return None
    
    @log_execution_time
    def get_training_data(
        self,
        feature_view_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        version: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training data from a feature view.
        
        Args:
            feature_view_name: Name of the feature view
            start_time: Start time for data (optional)
            end_time: End time for data (optional)
            version: Feature view version
        
        Returns:
            Tuple of (X_train, y_train) or (DataFrame, None) if no labels
        """
        if not self.is_connected():
            logger.error("Not connected to Hopsworks")
            return None, None
        
        try:
            fv = self._fs.get_feature_view(name=feature_view_name, version=version)
            
            if start_time and end_time:
                X, y = fv.training_data(
                    start_time=start_time,
                    end_time=end_time
                )
            else:
                X, y = fv.training_data()
            
            logger.info(f"Retrieved training data: {len(X)} samples")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None, None
    
    def close(self):
        """Close connection to Hopsworks."""
        if self._connection:
            # Connection cleanup is handled automatically
            self._connection = None
            self._project = None
            self._fs = None
            logger.info("Closed Hopsworks connection")


class FeatureStoreManager:
    """
    High-level manager for feature store operations.
    
    This class provides simplified interfaces for common
    feature store operations without requiring deep knowledge
    of Hopsworks internals.
    
    Usage:
        >>> manager = FeatureStoreManager()
        >>> manager.initialize()
        >>> manager.ingest_features(df)
        >>> X_train, y_train = manager.get_training_data()
    """
    
    # Feature groups we maintain
    FEATURE_GROUPS = {
        "aqi_features": {
            "description": "AQI and pollutant features",
            "primary_key": ["city", "timestamp"],
            "event_time": "timestamp"
        },
        "weather_features": {
            "description": "Weather-related features",
            "primary_key": ["city", "timestamp"],
            "event_time": "timestamp"
        }
    }
    
    def __init__(self, use_local_cache: bool = True):
        """
        Initialize feature store manager.
        
        Args:
            use_local_cache: Whether to cache features locally
        """
        self.client = HopsworksClient()
        self.use_local_cache = use_local_cache
        
        # Local cache directory
        settings = get_settings()
        self.cache_dir = Path(settings.app.features_dir) / "cache"
        if use_local_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._connected = False
    
    def initialize(self) -> bool:
        """
        Initialize connection and set up feature groups.
        
        Returns:
            True if initialization successful
        """
        # If no API key, use local mode
        if not self.client.api_key:
            logger.info("Running in local mode (no Hopsworks API key)")
            return True
        
        self._connected = self.client.connect()
        
        if self._connected:
            logger.info("Feature store initialized in cloud mode")
        else:
            logger.info("Failed to connect, running in local mode")
        
        return True  # Always return True, we can fall back to local
    
    @log_execution_time
    def ingest_features(
        self,
        df: pd.DataFrame,
        feature_group: str = "aqi_features"
    ) -> bool:
        """
        Ingest features into the feature store.
        
        Handles both cloud (Hopsworks) and local modes.
        
        Args:
            df: DataFrame with features
            feature_group: Target feature group name
        
        Returns:
            True if ingestion successful
        """
        logger.info(f"Ingesting {len(df)} rows to {feature_group}")
        
        # Always save locally for backup
        if self.use_local_cache:
            self._save_local(df, feature_group)
        
        # If connected, also push to Hopsworks
        if self._connected:
            config = self.FEATURE_GROUPS.get(feature_group, {})
            self.client.create_feature_group(
                df=df,
                name=feature_group,
                description=config.get("description"),
                primary_key=config.get("primary_key"),
                event_time=config.get("event_time")
            )
        
        return True
    
    def _save_local(self, df: pd.DataFrame, feature_group: str):
        """Save features locally as parquet."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.cache_dir / f"{feature_group}_{timestamp}.parquet"
        df.to_parquet(filepath, index=False)
        logger.debug(f"Saved local cache: {filepath}")
    
    @log_execution_time
    def get_training_data(
        self,
        feature_group: str = "aqi_features",
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get training data from feature store.
        
        Args:
            feature_group: Feature group to read from
            days_back: How many days of historical data
        
        Returns:
            DataFrame with features
        """
        # Try Hopsworks first
        if self._connected:
            try:
                fv_name = f"{feature_group}_view"
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                
                X, _ = self.client.get_training_data(
                    feature_view_name=fv_name,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if X is not None and len(X) > 0:
                    return X
            except Exception as e:
                logger.warning(f"Failed to fetch from Hopsworks: {e}")
        
        # Fall back to local cache
        return self._load_local(feature_group)
    
    def _load_local(self, feature_group: str) -> pd.DataFrame:
        """Load latest features from local cache."""
        pattern = f"{feature_group}_*.parquet"
        files = sorted(self.cache_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No local cache found for {feature_group}")
            return pd.DataFrame()
        
        # Load most recent file
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded {len(df)} rows from local cache: {latest_file.name}")
        
        return df
    
    def backfill_historical_data(
        self,
        df: pd.DataFrame,
        feature_group: str = "aqi_features"
    ) -> bool:
        """
        Backfill historical data into feature store.
        
        Used for initial loading of historical training data.
        
        Args:
            df: Historical data DataFrame
            feature_group: Target feature group
        
        Returns:
            True if backfill successful
        """
        logger.info(f"Backfilling {len(df)} historical records to {feature_group}")
        
        # Ensure timestamp is sorted
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        
        return self.ingest_features(df, feature_group)
    
    def close(self):
        """Close feature store connections."""
        if self._connected:
            self.client.close()
            self._connected = False


# Convenience function
def get_feature_store() -> FeatureStoreManager:
    """Get a configured feature store manager."""
    manager = FeatureStoreManager()
    manager.initialize()
    return manager


if __name__ == "__main__":
    # Test feature store
    print("Testing Feature Store Module")
    print("=" * 50)
    
    # Create sample data
    from src.data.feature_engineering import generate_synthetic_training_data
    
    print("\n1. Generating sample data:")
    df = generate_synthetic_training_data(n_samples=100)
    print(f"   Generated {len(df)} samples")
    
    # Initialize feature store
    print("\n2. Initializing feature store:")
    manager = FeatureStoreManager()
    manager.initialize()
    print("   Mode: Local (no Hopsworks API key)")
    
    # Ingest features
    print("\n3. Ingesting features:")
    manager.ingest_features(df, "aqi_features")
    print("   Features ingested successfully")
    
    # Retrieve features
    print("\n4. Retrieving training data:")
    training_data = manager.get_training_data("aqi_features")
    print(f"   Retrieved {len(training_data)} samples")
    
    # Cleanup
    manager.close()
    
    print("\n" + "=" * 50)
    print("Feature Store Test Complete!")
