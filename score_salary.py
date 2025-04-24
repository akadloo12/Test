import requests
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn  

def download_model():
    url = 'https://raw.githubusercontent.com/akadloo12/Test/main/salary_prediction_model_2025.pkl'
    local_filename = url.split('/')[-1]
    response = requests.get(url)
    open(local_filename, 'wb').write(response.content)

def is_model_found(file):
    model_path = Path(file)
    if not model_path.is_file():
        st.write(f"DEBUG: File `{model_path.absolute()}` not found. Let's download it! ⬇️")
        download_model()
    else:
        st.write(f"DEBUG: File `{model_path.absolute()}` found! 😎")

# Ensure the model is present
model_filename = "salary_prediction_model_2025.pkl"
is_model_found(model_filename)

# Load the saved model
model = pickle.load(open(model_filename, 'rb'))

# Streamlit UI
st.title("💼 Data Scientist Salary Prediction")

education = st.selectbox("Highest Level of Education", [
    "I never completed any formal education",
    "Primary/elementary school",
    "Some college/university study without earning a bachelor’s degree",
    "Associate degree",
    "Bachelor’s degree",
    "Master’s degree",
    "Doctoral degree",
    "Professional degree"
])

years_coding = st.selectbox("Years of Coding Experience", [
    "I have never written code",
