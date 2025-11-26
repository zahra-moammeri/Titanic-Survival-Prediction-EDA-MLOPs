
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")


class AgeFareLog(BaseEstimator, TransformerMixin):
    """Calculating Age and Fare features' logs in two different features of 
    'AgeLog' and 'FareLog', and filling nulls of them.
    Filling null values of Age feature based on 'Sex' and 'Pclass' median.
    Also filling Fare null values with mean of them
    """
    def __init__(self, age = "Age", age_log= "AgeLog", pclass = "Pclass", 
                 sex = "Sex", fare = "Fare", fare_log = "FareLog"):
        self.age = age
        self.age_log = age_log
        self.pclass = pclass
        self.sex = sex
        self.fare = fare
        self.fare_log = fare_log

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        new_data = X.copy()
        
        new_data[self.age] = new_data[self.age].fillna(new_data.groupby([self.sex, self.pclass])[self.age].transform("median"))
        new_data[self.age] = new_data[self.age].fillna(new_data[self.age].median())
        new_data[self.fare] = new_data[self.fare].fillna(new_data[self.fare].mean())

        new_data[self.age_log] = np.log1p(new_data[self.age])
        new_data[self.fare_log] = np.log1p(new_data[self.fare])
        # new_data.drop(columns=[self.age, self.fare], inplace=True)

        return new_data


class HasFamily(BaseEstimator, TransformerMixin):
    """If the passengers have family based on the 'SibSp' and 'Parch'.
    1 -> If the passenger has sibling, spouse, parent or children.
    0 -> If the passenger is alone.
    """
    def __init__(self, sibsp="SibSp", parch="Parch", has_family="HasFamily"):
        self.sibsp = sibsp
        self.parch = parch
        self.has_family = has_family

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame):
        new_data = X.copy()
        family_size =  X[self.sibsp] + X[self.parch]
        new_data[self.has_family] = family_size.fillna(0)
        new_data[self.has_family] = family_size.apply(lambda x: 1 if x>=1 else 0).fillna(0)
        # new_data.drop(columns=[self.sibsp, self.parch], inplace=True)

        return new_data


class HasCabin(BaseEstimator, TransformerMixin):
    """Check if the passenger has a 'Cabin' 
    If 'Cabin' is null fill it with 0."""
    def __init__(self, cabin="Cabin", has_cabin="HasCabin"):
        self.cabin = cabin
        self.has_cabin = has_cabin

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame):
        new_data = X.copy()
        new_data[self.cabin] = new_data[self.cabin].astype("str")
        new_data[self.cabin] = new_data[self.cabin].apply(lambda x: x.replace(x, x[0]) if x not in "nan" else "N")
        new_data[self.cabin] = new_data[self.cabin].fillna("N")
        new_data[self.has_cabin] = new_data[self.cabin].apply(lambda x: 0 if x == "N" else 1).fillna(0)
        

        return new_data 


class PclassImputer(BaseEstimator, TransformerMixin):
    """Fill in the 'Pclass' null values. 
    Passenger Class depended on 'Sex' and 'Cabin Type', so I fill it
    based on the sex and HasCabin grouping and the 'most frequent' of them.
    'Sex' feature also filled with most frequent one.
    """
    def __init__(self, pclass="Pclass", sex="Sex", has_cabin="HasCabin"):
        self.pclass = pclass
        self.sex = sex
        self.has_cabin = has_cabin

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        new_data = X.copy()
        new_data[self.sex] = new_data[self.sex].fillna(new_data[self.sex].mode().iloc[0])
        new_data[self.has_cabin] = new_data[self.has_cabin].fillna(0)
        new_data[self.pclass] = new_data[self.pclass].fillna(
            new_data.groupby([self.sex, self.has_cabin])[self.pclass].transform(
                lambda x: x.mode().iloc[0] if any(x.mode()) else np.nan
            ))
        new_data[self.pclass] = new_data[self.pclass].fillna(new_data[self.pclass].mode().iloc[0])

        return new_data


class EncoderTransform(BaseEstimator, TransformerMixin):
    """Convert Categorical features ('Sex' and 'Embarked) to numerical format
    using OneHotEncoder."""
    def __init__(self, sex="Sex", embarked="Embarked"):
        self.sex = sex
        self.embarked = embarked

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame):
        new_data = X.copy()
        new_data[self.sex] = new_data[self.sex].fillna(new_data[self.sex].mode()[0])
        new_data[self.embarked] = new_data[self.embarked].fillna(
                                                    new_data[self.embarked].mode()[0])

        new_data[self.embarked] = OneHotEncoder(drop="first", 
                                                sparse_output=False).fit_transform(new_data[[self.embarked]])
        new_data[self.sex] = OneHotEncoder(drop="first", 
                                           sparse_output=False).fit_transform(new_data[[self.sex]])

        return new_data


# class DropColumns(BaseEstimator, TransformerMixin):
#     """Drop other columns not used in prediction"""
#     def __init__(self, name="Name", ticket="Ticket", 
#                 #  sibsp="SibSp",
#     #              parch="Parch", cabin="Cabin", fare="Fare", age="Age"
#     ):
#         self.name = name
#         self.ticket=ticket
#         # self.sibsp= sibsp
#         # self.parch = parch
#         # self.cabin = cabin
#         # self.fare = fare
#         # self.age = age

#     def fit(self, X:pd.DataFrame, y=None):
#         return self

#     def transform(self, X:pd.DataFrame):
#         new_data = X.copy()
#         new_data.drop([self.name, self.ticket, 
#                     #    self.sibsp, self.parch, 
#                     #    self.cabin, self.fare, self.age
#                        ], inplace=True, axis=1)
#         return new_data


# class DebugTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, name):
#         self.name = name
    
#     def fit(self, X, y=None):
#         print(f"\n=== {self.name} - FIT ===")
#         print(f"Shape: {X.shape}")
#         print(f"Columns: {X.columns.tolist()}")
#         return self
    
#     def transform(self, X):
#         print(f"\n=== {self.name} - TRANSFORM ===")
#         print(f"Shape: {X.shape}")
#         print(f"Columns: {X.columns.tolist()}")
#         return X