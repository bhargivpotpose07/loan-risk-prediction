import streamlit as st
import pandas as pd
from model import train_model

st.title("🏦 Loan Risk Prediction System (Advanced)")

model, columns = train_model()

st.subheader("📋 Fill Loan Application")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Applicant Income", value=5000)
co_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount", value=150)
loan_term = st.number_input("Loan Term", value=360)
credit = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    # Fill numeric
    input_data["ApplicantIncome"] = income
    input_data["CoapplicantIncome"] = co_income
    input_data["LoanAmount"] = loan_amount
    input_data["Loan_Amount_Term"] = loan_term
    input_data["Credit_History"] = credit

    # Encoding helper
    def set_col(name):
        if name in input_data.columns:
            input_data[name] = 1

    set_col(f"Gender_{gender}")
    set_col(f"Married_{married}")
    set_col(f"Dependents_{dependents}")
    set_col(f"Education_{education}")
    set_col(f"Self_Employed_{self_employed}")
    set_col(f"Property_Area_{property_area}")

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected (Risk: {probability[0][0]*100:.2f}%)")