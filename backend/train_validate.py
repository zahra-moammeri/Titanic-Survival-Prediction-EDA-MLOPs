import pandas as pd
import joblib
from pathlib import Path
from model_scripts.utils import read_data, save_model
from model_scripts import calculate_metrics, preprocessing
from model_scripts.preprocessing import logistic_clf


def train_all_models(data_path="data/titanic.csv"):

    X_train, _, y_train, _ = read_data(path=data_path)

    results = {}
    for clf in preprocessing.classifiers:
        class_name, metrics = calculate_metrics.model_evaluate(clf, X_train, y_train)
        results[class_name] = metrics
        save_model(clf, "models/experienced_models")

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    
    print("metrics evaluation of training data: ")
    print("="*100)
    print(metrics_df)
    print("="*100)


def validate_all_models(data_path="data/titanic.csv"):
    _, X_val, _, y_val = read_data(path=data_path)

    val_results = {}
    for clf in preprocessing.classifiers:
        class_name, metrics = calculate_metrics.model_predict(clf, X_val, y_val)
        val_results[class_name] = metrics

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(val_results, orient="index")

    print("metrics evaluation of validation data: ")
    print("="*100)
    print(metrics_df)
    print("="*100)


def train_model(data_path="data/titanic.csv", path_to_save="models"):
    path = Path(path_to_save)
    path.mkdir(parents=True, exist_ok=True)
    
    X_train, _, y_train, _ = read_data(path=data_path)
    logistic_clf.fit(X_train, y_train)
    
    filename = path / "logistic_model.joblib"
    joblib.dump(logistic_clf, filename)
    print(f"Model saved successfully at {filename}")
    return filename


if __name__ == "__main__":
    train_all_models()
    validate_all_models()
    # train_model()