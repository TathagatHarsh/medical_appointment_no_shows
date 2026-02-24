import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="No-Show Prediction", layout="centered")
st.title("Clinical Appointment No-Show Prediction")

# Load trained model (we'll save this next)
model = joblib.load("model.joblib")

st.subheader("Enter Patient Details")

gender = st.selectbox("Gender", ["F", "M"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
scholarship = st.selectbox("Scholarship", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
alcoholism = st.selectbox("Alcoholism", [0, 1])
handcap = st.selectbox("Handicap", [0, 1])
sms_received = st.selectbox("SMS Received", [0, 1])
waiting_days = st.number_input("Waiting Days", min_value=0, max_value=365, value=5)
neighbourhood = st.text_input("Neighbourhood", "JARDIM DA PENHA")

if st.button("Predict No-Show"):
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Neighbourhood": neighbourhood,
        "Scholarship": scholarship,
        "Hipertension": hypertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "WaitingDays": waiting_days
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely No-Show (probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to Show (probability: {1-prob:.2f})")
