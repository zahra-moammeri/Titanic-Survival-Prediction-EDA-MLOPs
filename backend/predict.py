import pandas as pd
import joblib

def predict_new_data(new_data:pd.DataFrame, model_path="models/logistic_model.joblib"):
    model = joblib.load(model_path)
    predictions = model.predict(new_data)
    return predictions



if __name__ == "__main__":
    X_test = pd.DataFrame([{'Pclass': 3.,'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male',
                'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.2500,
                'Cabin':'NaN', 'Embarked':'S'}])

    print(predict_new_data(new_data=X_test)[0])