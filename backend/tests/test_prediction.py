import pytest
import joblib
import numpy as np
from backend.model_scripts.utils import save_model, load_model
from backend.model_scripts.preprocessing import classifiers


class TestPredictions:
    
    def test_model_saving_loading(self):

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        simple_clf = Pipeline([("classifier", LogisticRegression())])
        # Test Saving
        save_model(simple_clf)
        # Test Loading
        loaded_model = load_model(simple_clf)
        assert loaded_model is not None

    def test_prediction_consistency(self, sample_train_data):
        X_train, y_train = sample_train_data

        for clf in classifiers:
            clf.fit(X_train, y_train)

            pred1 = clf.predict(X_train)
            pred2 = clf.predict(X_train)

            assert np.array_equal(pred1, pred2)

    def test_prediction_correctness(self, sample_test_data):
        for clf in classifiers:
            pred = clf.predict(sample_test_data)
            assert set(pred).issubset({0, 1}), "prediction must be 0 or 1"