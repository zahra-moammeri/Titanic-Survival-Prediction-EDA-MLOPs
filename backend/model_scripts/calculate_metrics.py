
import time
import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from model_scripts.utils import load_model

import warnings
warnings.filterwarnings("ignore")


def scores(classifier, X, y, cv):
    acc = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy").mean()
    roc = cross_val_score(classifier, X, y, cv=cv, scoring="roc_auc").mean()
    y_pred = cross_val_predict(classifier, X, y, cv=cv)
    precision = precision_score(y, y_pred) * 100
    recall = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100

    return acc, roc, precision, recall, f1


def model_evaluate(model_name, X_train, y_train):
    """Calculate training Scores and Run Time"""
    start_time = time.time()
    model_name.fit(X_train, y_train)
    # acc, roc, precision, recall, f1 = scores(model_name, X_train, y_train, cv=5)
    y_pred = model_name.predict(X_train)
    y_proba = model_name.predict_proba(X_train)[:, 1] if hasattr(model_name, "predict_proba") else None

    acc = accuracy_score(y_train, y_pred)
    roc = roc_auc_score(y_train, y_proba) if y_proba is not None else np.nan
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    
    end_time = time.time()
    class_name = model_name["classifier"].__class__.__name__
    metrics = {"accuracy": np.round(acc*100, 3), 
               "roc": np.round(roc*100, 3),
               "precision": np.round(precision, 3),
               "recall": np.round(recall, 3),
               "f1 score": np.round(f1, 3),
               "Running time": np.round(end_time-start_time, 4)}
    return class_name, metrics


def model_predict(model, X_test, y_test):
    """Calculate prediction Scores"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    end_time = time.time()

    class_name = model["classifier"].__class__.__name__
    metrics = {"precision": np.round(precision*100, 3),
               "recall": np.round(recall*100, 3),
               "f1 score": np.round(f1*100, 3),
               "Running time": np.round(end_time-start_time, 4)}

    return class_name, metrics
