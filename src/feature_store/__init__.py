"""
Feature Store Module
====================

Integration with Hopsworks Feature Store for feature management.
"""

from src.feature_store.hopsworks_client import HopsworksClient, FeatureStoreManager

__all__ = ["HopsworksClient", "FeatureStoreManager"]
