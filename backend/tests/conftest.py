import sys
import os
import pytest
import pandas as pd
import numpy as np
from backend.model_scripts.utils import read_data
from fastapi.testclient import TestClient
from backend.main import app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Test client for FastAPI"""
    yield TestClient(app)


@pytest.fixture
def data():
    X_train, X_val, y_train, y_val = read_data()
    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_val": X_val,
        "y_val": y_val
    }


@pytest.fixture
def sample_train_data():
    X_train = pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Age': [25., 42., 28., 35., 57., 22., 31., 45., 19., 60., 33., 29.],
        'Fare': [10.0, 120.0, 30.0, 45.0, 87.0, 15.0, 60.0, 200.0, 8.0, 150.0, 75.0, 25.0],
        'Sex': ['female', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'Name': ['Sarah', 'Ali', 'Mina', 'Dr', 'Hadi', 'Lily', 'John', 'Emma', 'Mike', 'Anna', 'Tom', 'Lisa'],
        'Ticket': [1563, 15678, 8674, 4531, 8565, 9234, 6745, 3456, 7823, 4567, 8934, 5678],
        'Pclass': [2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3],
        'SibSp': [0, 1, 0, 2, 3, 1, 0, 1, 2, 0, 1, 0],
        'Parch': [0, 1, 2, 0, 4, 1, 0, 2, 1, 0, 1, 0],
        'Cabin': ['A1', None, 'B2', 'C3', 'F21', 'D4', None, 'E5', 'G6', 'H7', None, 'I8'], 
        'Embarked': ['S', 'Q', 'S', 'C', 'C', 'Q', 'S', 'C', 'Q', 'S', 'C', 'Q']
    })
    y_train = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # Balanced classes
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    X_test = pd.DataFrame([{'Pclass': 3.,'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male',
                'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.2500,
                'Cabin':'NaN', 'Embarked':'S'}])
    
    return X_test