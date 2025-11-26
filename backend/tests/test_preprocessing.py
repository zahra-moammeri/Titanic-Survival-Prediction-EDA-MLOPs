import pytest
import pandas as pd
import warnings
from backend.model_scripts.preprocessing import preprocess_pipeline, classifiers


class TestPreprocessing:
    
    # @pytest.fixture
    # def sample_data(self):
    #     sample_df = pd.DataFrame({
    #         'PassengerId': [1, 2, 3, 4],
    #         'Age': [25., 30., None, 35.],
    #         'Fare': [10.0, 20.0, 30.0, None],
    #         'Sex': ['male', 'female', 'male', 'female'],
    #         'Name':['Sarah', 'Ali', 'Mina', 'Dr'],
    #         'Ticket':[1563, 15678, 8674, 4531],
    #         'Pclass': [1, 2, 3, 1],
    #         'SibSp': [0, 1, 0, 2],
    #         'Parch': [0, 1, 2, 0],
    #         'Cabin': ['A1', None, 'B2', 'C3'], 
    #         'Embarked': ['S', 'Q', 'S', 'C']
    #     })
    #     return sample_df
    
    def test_pipeline_steps(self):
        expected_steps = ['has_family',
                        'pclass_hascabin',
                        'age_fare_log',
                        'sex_embark_encode',
                        ]
        assert list(preprocess_pipeline.named_steps.keys()) == expected_steps

    def test_all_classifiers_have_pipeline(self):
        for clf in classifiers:
            assert "preprocessing" in clf.named_steps
            assert "classifier" in clf.named_steps

    def test_pipeline_fit_predict(self, sample_train_data):
        X_sample, y_sample = sample_train_data
        for clf in classifiers:
            try:
                fitted_clf = clf.fit(X_sample, y_sample)
                print("Fitted Successfully...")
                assert hasattr(fitted_clf, "predict")
                predictions = fitted_clf.predict(X_sample)
                print("Predicted Successfully...")
                assert len(predictions) == len(y_sample)
                assert set(predictions).issubset({0 ,1})
            except Exception as e:
                print("Error", e)
                print("Error Type", type(e).__name__)
                raise
            