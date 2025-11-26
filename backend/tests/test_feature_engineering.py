import pytest
import pandas as pd
import numpy as np
from backend.model_scripts.feature_engineering import (AgeFareLog,
                                                HasFamily,
                                                PclassImputer,
                                                HasCabin)

class TestFeatureEngineering:
    
    def test_age_fare_log_transformer(self, sample_train_data):

        X_train, y_train = sample_train_data
        transformer = AgeFareLog()
        trans_data = transformer.fit_transform(X_train)
        # Teat new feature created
        assert "AgeLog" in trans_data.columns
        assert "FareLog" in trans_data.columns
        # Test null values
        assert trans_data["AgeLog"].isnull().sum() == 0
        assert trans_data["FareLog"].isnull().sum() == 0
        # Test Log Transformed
        assert all(trans_data["AgeLog"] == np.log1p(trans_data["Age"].fillna(
                                                trans_data["Age"].median())))
        assert all(trans_data["FareLog"] == np.log1p(trans_data["Fare"].fillna(
                                                trans_data["Fare"].mean())))


    def test_has_family_transformer(self, sample_train_data):

        X_train, _ = sample_train_data
        transformer = HasFamily()
        trans_data = transformer.fit_transform(X_train)
        # Test new feature created
        assert "HasFamily" in trans_data.columns
        # Test null values
        assert trans_data["HasFamily"].isnull().sum() == 0
        # Test subsets
        assert set(trans_data["HasFamily"].unique()).issubset({0, 1})


    def test_has_cabin_transformer(self, sample_train_data):

        X_train, _ = sample_train_data
        transformer = HasCabin()
        trans_data = transformer.fit_transform(X_train)
        #Test new feature created
        assert "HasCabin" in trans_data.columns
        # Test null values
        assert trans_data["HasCabin"].isnull().sum() == 0
        # Test Subsets in {0, 1}
        assert set(trans_data["HasCabin"].unique()).issubset({0, 1})


    def test_pclass_transformer(self):
        sample_df = pd.DataFrame({
            "Pclass": [1, 1, np.nan, 1, 3, np.nan, np.nan, 2],
            "Sex": ["Female", np.nan, "Female", "Female", "Female", "Female", "Male", "Male"],
            "HasCabin": [1, 0, 0, 1, 1, np.nan, 0, 1]
        })
        transformer = PclassImputer()
        trans_data = transformer.fit_transform(sample_df)
        # Test null values 
        assert trans_data["Pclass"].isnull().sum() == 0
        assert trans_data["Sex"].isnull().sum() == 0
        assert trans_data["HasCabin"].isnull().sum() == 0

