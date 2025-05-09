import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💓 Heart Failure Risk Predictor")

st.write("Enter patient details to predict risk of death due to heart failure.")

# Input form
with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0.0, value=250.0)
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0.0, max_value=100.0, value=40.0)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=250000.0)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.2)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0.0, value=137.0)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    time = st.number_input("Follow-up Period (days)", min_value=0, value=100)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = np.array([
        age, anaemia, creatinine_phosphokinase, diabetes,
        ejection_fraction, high_blood_pressure, platelets,
        serum_creatinine, serum_sodium, sex, smoking, time
    ]).reshape(1, -1)

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ Patient will likely **die**.")
    else:
        st.success("✅ Patient will likely **survive**.")
