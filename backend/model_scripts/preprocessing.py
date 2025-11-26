
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from model_scripts.feature_engineering import (HasFamily, HasCabin, PclassImputer, AgeFareLog)

import warnings
warnings.filterwarnings("ignore")


pclass_cabin_pipeline = Pipeline([
    ("has_cabin", HasCabin()),
    ("pclass_imputer", PclassImputer())
])

sex_embark_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

sex_embark = ColumnTransformer(transformers=[
    ("sex_embark", sex_embark_pipeline, ["Sex", "Embarked"]),
    ("pass_through", "passthrough", ["Pclass", "HasFamily", "HasCabin", "AgeLog", "FareLog"]),
])

preprocess_pipeline = Pipeline([
    ("has_family", HasFamily()),
    ("pclass_hascabin", pclass_cabin_pipeline),
    ("age_fare_log", AgeFareLog()),
    ("sex_embark_encode", sex_embark),
], verbose=False)


logistic_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", LogisticRegression(random_state=42))
])

sgd_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", SGDClassifier(random_state=42))
])

svc_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", SVC(random_state=42))
])

decision_tree_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

random_forest_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", RandomForestClassifier(random_state=42))
])

naive_bayes_clf = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", GaussianNB())
])

classifiers = [logistic_clf, sgd_clf, svc_clf, decision_tree_clf, random_forest_clf, naive_bayes_clf]