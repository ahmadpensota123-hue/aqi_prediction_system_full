"""
ML Validation Tests using DeepChecks
=====================================

This module uses DeepChecks to validate:
- Data integrity
- Data drift detection
- Model performance checks
- Feature importance validation
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_engineering import (
    generate_synthetic_training_data,
    FeatureEngineer,
)

# Try to import DeepChecks
try:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import (
        DataDuplicates,
        MixedNulls,
        FeatureLabelCorrelation,
        DatasetsSizeComparison,
    )
    from deepchecks.tabular.suites import data_integrity

    HAS_DEEPCHECKS = True
except ImportError:
    HAS_DEEPCHECKS = False


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    df = generate_synthetic_training_data(n_samples=500)

    fe = FeatureEngineer()
    df = fe.add_lag_features(df)
    df = fe.add_rolling_features(df)
    df = fe.add_derived_features(df)
    df = df.dropna()

    # Prepare features
    exclude = [
        "timestamp",
        "city",
        "aqi_category_name",
        "aqi_category",
        "dominant_pollutant",
        "weather_condition",
    ]

    features = [c for c in df.columns if c not in exclude and c != "aqi"]

    return df, features


@pytest.mark.skipif(not HAS_DEEPCHECKS, reason="DeepChecks not installed")
class TestDataIntegrity:
    """Data integrity tests using DeepChecks."""

    def test_no_excessive_duplicates(self, sample_dataset):
        """Check that data doesn't have excessive duplicate rows."""
        df, features = sample_dataset

        ds = Dataset(df[features], label=df["aqi"], cat_features=[])
        check = DataDuplicates()
        result = check.run(ds)

        # Allow up to 5% duplicates
        assert result.value["percent"] < 5, "Too many duplicate rows in data"

    def test_feature_label_correlation(self, sample_dataset):
        """Check that features have correlation with label."""
        df, features = sample_dataset

        ds = Dataset(df[features], label=df["aqi"], cat_features=[])
        check = FeatureLabelCorrelation()
        result = check.run(ds)

        # At least some features should correlate with label
        correlations = list(result.value.values())
        max_correlation = max([abs(c) for c in correlations if c is not None] or [0])

        assert max_correlation > 0.1, "No features correlate with label"


class TestDataQuality:
    """Basic data quality tests (no DeepChecks required)."""

    def test_no_all_null_columns(self, sample_dataset):
        """Check that no columns are entirely null."""
        df, features = sample_dataset

        for col in features:
            null_pct = df[col].isnull().sum() / len(df)
            assert null_pct < 1.0, f"Column {col} is entirely null"

    def test_aqi_values_valid(self, sample_dataset):
        """Check that AQI values are in valid range."""
        df, _ = sample_dataset

        assert df["aqi"].min() >= 0, "Negative AQI values found"
        assert df["aqi"].max() <= 600, "AQI values exceed valid range"

    def test_feature_variance(self, sample_dataset):
        """Check that features have non-zero variance."""
        df, features = sample_dataset

        for col in features[:20]:  # Check first 20 features
            if df[col].dtype in [np.float64, np.int64]:
                variance = df[col].var()
                assert variance > 0, f"Feature {col} has zero variance"

    def test_no_infinite_values(self, sample_dataset):
        """Check for infinite values in numeric columns."""
        df, features = sample_dataset

        numeric_cols = df[features].select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"Infinite values found in {col}"


class TestModelInputValidation:
    """Validate model inputs."""

    def test_feature_count(self, sample_dataset):
        """Check that we have sufficient features."""
        df, features = sample_dataset

        assert len(features) >= 10, "Too few features for model training"

    def test_sample_count(self, sample_dataset):
        """Check that we have sufficient samples."""
        df, _ = sample_dataset

        assert len(df) >= 100, "Too few samples for model training"

    def test_train_test_split_possible(self, sample_dataset):
        """Check that data can be split for training/testing."""
        df, _ = sample_dataset

        # Need at least 50 samples for 80/20 split
        assert len(df) >= 50, "Not enough data for train/test split"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
