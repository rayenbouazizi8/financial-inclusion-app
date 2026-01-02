import streamlit as st
import pandas as pd
import joblib

model = joblib.load("financial_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💰 Financial Inclusion Prediction")

age = st.slider("Age of Respondent", 16, 100, 30)
household_size = st.slider("Household Size", 1, 20, 3)
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["No formal education", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other"])
job_type = st.selectbox("Job Type", ["Self employed", "Government Dependent", "Formally employed Private", "Informally employed", "Formally employed Government", "Farming and Fishing", "Remittance Dependent", "Other Income", "No Income", "Dont Know"])

# Convert inputs to dataframe with dummy columns (must match training)
input_dict = {
    "age_of_respondent": age,
    "household_size": household_size,
    "cellphone_access_Yes": 1 if cellphone_access == "Yes" else 0,
    "location_type_Urban": 1 if location_type == "Urban" else 0,
    "gender_of_respondent_Male": 1 if gender == "Male" else 0,
    "education_level_Primary education": 1 if education == "Primary education" else 0,
    "education_level_Secondary education": 1 if education == "Secondary education" else 0,
    "education_level_Tertiary education": 1 if education == "Tertiary education" else 0,
    "education_level_Vocational/Specialised training": 1 if education == "Vocational/Specialised training" else 0,
    "education_level_Other": 1 if education == "Other" else 0,
    "job_type_Self employed": 1 if job_type == "Self employed" else 0,
    "job_type_Informally employed": 1 if job_type == "Informally employed" else 0,
    "job_type_Formally employed Private": 1 if job_type == "Formally employed Private" else 0,
    "job_type_Formally employed Government": 1 if job_type == "Formally employed Government" else 0,
    "job_type_Government Dependent": 1 if job_type == "Government Dependent" else 0,
    "job_type_Remittance Dependent": 1 if job_type == "Remittance Dependent" else 0,
    "job_type_Farming and Fishing": 1 if job_type == "Farming and Fishing" else 0,
    "job_type_Other Income": 1 if job_type == "Other Income" else 0,
    "job_type_Dont Know": 1 if job_type == "Dont Know" else 0
}

# Fill in all columns model expects
model_features = pd.read_csv("financial_cleaned.csv").drop("HasBankAccount", axis=1).columns
input_df = pd.DataFrame([input_dict])
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

if st.button("Predict"):
    st.subheader("Prediction")
    if prediction == 1:
        st.success("✅ Likely to HAVE a Bank Account")
    else:
        st.error("❌ Unlikely to Have a Bank Account")
