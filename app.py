import streamlit as st
import pandas as pd
import joblib
@st.cache_data
def load_neighbourhoods():
    df = pd.read_csv("medical_appointment_no_shows.csv")  # use your actual filename
    return sorted(df["Neighbourhood"].unique().tolist())

neighbourhoods = load_neighbourhoods()

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
neighbourhood = st.selectbox("Neighbourhood", neighbourhoods, index=0)

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

st.caption("⚠️ This is a demo ML model for academic purposes. Predictions are not medical advice.")
with st.expander("How this works"):
    st.write("""
    The model uses patient demographics and appointment details 
    (e.g., age, waiting days, SMS reminders) to predict the likelihood of a no-show.
    """)