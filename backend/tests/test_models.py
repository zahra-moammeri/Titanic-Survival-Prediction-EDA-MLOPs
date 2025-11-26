import pytest
import numpy as np
import pandas as pd

from backend.model_scripts.calculate_metrics import model_evaluate
from backend.model_scripts.preprocessing import classifiers


class TestModel:

    def test_evaluation_metrics(self, sample_train_data):
        X_train, y_train = sample_train_data
        
        for clf in classifiers:
            class_name, metrics = model_evaluate(clf, X_train, y_train)
            # Test types
            assert isinstance(class_name, str), "Class name is not 'string'"
            assert isinstance(metrics, dict), "Metrics are not in a dictionary"
           
            # Test expected metrics
            expected_metrics = ["accuracy", "roc", 
                                "precision", "recall", 
                                "f1 score", "Running time"]
            for metric in expected_metrics:
                assert metric in metrics, f"'metric {metric}' is not in expected metrics list"

            # Test metrics range
            assert 0. <= metrics["accuracy"] <= 100., "'accuracy' is out of range"
            assert 0. <= metrics["precision"] <= 100., "'precision' is out of range"
            assert 0. <= metrics["recall"] <= 100., "'recall' is out of range"
            assert 0. <= metrics["f1 score"] <= 100., "'f1 score' is out of range"