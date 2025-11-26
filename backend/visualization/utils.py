
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc

from backend.model_scripts.utils import load_model

import warnings
warnings.filterwarnings("ignore")


def conf_matrix(model_name, X_train, y_train):
    path = "models/experienced_models"
    model = load_model(model_name= model_name, model_path=path)
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    conf_val = confusion_matrix(y_train, y_pred)
    class_name = model_name["classifier"].__class__.__name__
    return class_name, conf_val


def roc_curve_plots(model_name, X_train, y_train):
    path = "models/experienced_models"
    model = load_model(model_name= model_name, model_path=path)
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    class_name = model_name["classifier"].__class__.__name__
    return class_name, fpr, tpr, thresholds, roc_auc

