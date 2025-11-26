
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split



def read_data(path="data/titanic.csv", index_col="PassengerId", target="Survived"):
    train = pd.read_csv(path, index_col=index_col)
    X = train.drop(columns=[target], axis="columns")
    y = train[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    
    return X_train, X_val, y_train, y_val


def save_model(model_name, path_to_save="models/experienced_models"):
    path = Path(path_to_save)
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"{model_name['classifier'].__class__.__name__}.joblib"
    joblib.dump(model_name, filename)
    print(f"Model saved successfully at {filename}")


def load_model(model_name, model_path="models/experienced_models"):
    path = Path(model_path)
    if path.is_dir():
        filename = f"{model_name['classifier'].__class__.__name__}.joblib"
        try:
            filepath = path / filename
            model = joblib.load(filepath)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"There is no such file named {filename} ")
    else:
        raise  FileNotFoundError(f"There is no directory called {path}")
    