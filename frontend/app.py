# import os
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''
# os.environ['NO_PROXY'] = '127.0.0.1,localhost'

import streamlit as st
import requests

API_URL = "http://backend:8000/predict"

st.title("Titanic Survival Prediction")

name = st.text_input("Name")
sex = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", 0, 100)
ticket = st.text_input("Ticket number")
cabin = st.text_input("Cabin number")
pclass = st.selectbox("Ticket Class", [1, 2, 3])
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Queenstown", "Cherbourg"])
sibsp = st.number_input("No. of Siblings/Spouses aboard", 0, 50)
parch = st.number_input("No. of parents/children aboard", 0, 50)
fare = st.number_input("Fare")


if st.button("Predict"):
    payload = {
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked
        }
    
    try:    
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        if result["prediction"] == 0:
            st.warning(f"Unfortunately '{name}' did Not Survived.")
        elif result["prediction"] == 1:
            st.success(f"Fortunately '{name}' Survived.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")