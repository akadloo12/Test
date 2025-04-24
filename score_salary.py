import requests
from pathlib import Path
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn



def download_model():
    url = 'https://github.com/akadloo12/Test/blob/main/salary_prediction_model_2025.pkl'
    local_filename = url.split('/')[-1]
    response = requests.get(url)
    open(local_filename, 'wb').write(response.content)

def is_model_found(file):
    model_path = Path(file)
    found = model_path.is_file()
    if not found:
        st.write(f"DEBUG: File `{model_path.absolute()}` not found. Let's download it! :arrow_down:")
        download_model()
    else:
        st.write(f"DEBUG: File `{model_path.absolute()}` found! :sunglasses:")


model_filename = "salary_prediction_model_2025.pkl"
is_model_found(model_filename)
model = pd.read_pickle(model_filename)

# Load the saved model
model = pickle.load(open('salary_prediction_model_2025.pkl', 'rb'))

# Optional: Load your scaler if you used one
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit page title
st.title("💼 Data Scientist Salary Prediction")

# Input fields
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
    "I have never written code", "< 1 years", "1-3 years", "3-5 years",
    "5-10 years", "10-20 years", "20+ years"
])

country = st.selectbox("Country", [
    "United States", "India", "China", "Germany", "United Kingdom", "Canada",
    "France", "Brazil", "Other"
])

# Language checkboxes
languages = ['Python', 'R', 'SQL', 'C', 'C#', 'C++', 'Java', 'Javascript', 'Bash', 'PHP', 'MATLAB', 'Julia', 'Go', 'None', 'Other']
language_inputs = [1 if st.checkbox(lang) else 0 for lang in languages]

# Mapping for education
education_map = {
    "I never completed any formal education": 0,
    "Primary/elementary school": 0,
    "Some college/university study without earning a bachelor’s degree": 1,
    "Associate degree": 1,
    "Bachelor’s degree": 2,
    "Master’s degree": 3,
    "Doctoral degree": 4,
    "Professional degree": 4
}
edu_val = education_map[education]

# Mapping for coding experience
coding_map = {
    "I have never written code": 0,
    "< 1 years": 0.5,
    "1-3 years": 2,
    "3-5 years": 4,
    "5-10 years": 7.5,
    "10-20 years": 15,
    "20+ years": 25
}
code_val = coding_map[years_coding]

# Manual country one-hot encoding (example with a few + 'Other')
country_features = {
    "United States": 0,
    "India": 0,
    "China": 0,
    "Germany": 0,
    "United Kingdom": 0,
    "Canada": 0,
    "France": 0,
    "Brazil": 0,
    "Country_Other": 0
}
if country in country_features:
    country_features[country] = 1
else:
    country_features["Country_Other"] = 1

# Final input array
input_data = [edu_val, code_val] + language_inputs + list(country_features.values())

# Optional: scale input if you used a scaler during training
# numeric_data = scaler.transform([[edu_val, code_val]])
# input_data[:2] = numeric_data[0]

# Predict
if st.button("Predict Salary"):
    salary = model.predict([input_data])[0]
    st.success(f"💰 Estimated Salary: ${salary:,.2f} per year")
