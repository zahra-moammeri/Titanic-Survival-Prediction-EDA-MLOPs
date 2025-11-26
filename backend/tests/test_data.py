import pytest
import pandas as pd
from backend.model_scripts.utils import read_data


class TestDataValidation:
    
    def test_data_shape(self, data):
        assert data["x_train"].shape[0] == data["y_train"].shape[0]
        assert data["x_val"].shape[0] == data["y_val"].shape[0]

    def test_data_columns(self, data):
        expected_columns = ["Pclass", "Name", "Sex", "Age",	"SibSp", 
                            "Parch", "Fare", "Cabin",	"Embarked"]
        for col in expected_columns:
            assert col in data["x_train"].columns

    def test_target_subsets(self, data):
        assert set(data["y_train"].unique()).issubset({0, 1})

    def test_target_isnull(self, data):
        assert data["y_train"].isnull().sum() == 0

    def test_data_leakage(self, data):
        train_indices = set(data["x_train"].index)
        val_indices = set(data["x_val"].index)
        assert len(train_indices.intersection(val_indices)) == 0

    def test_data_type(self):
        X_train, X_val, y_train, y_val = read_data()
        assert isinstance(X_train, pd.DataFrame), "'X_train' msut be a pandas DataFrame"
        assert isinstance(X_val, pd.DataFrame), "'X_val' is must be pandas DataFrame"
        assert isinstance(y_train, pd.Series), "'y_train' must be a pandas Series"
        assert isinstance(y_val, pd.Series), "'y_val' must be a pandas Series"

