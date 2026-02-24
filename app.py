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
st.write("This app predicts whether a patient will show up or miss their appointment based on demographic and scheduling features.")

# Load trained model (we'll save this next)
model = joblib.load("model.joblib")

st.subheader("Enter Patient Details")

def yes_no_to_int(label):
    return 1 if label == "Yes" else 0

gender = st.selectbox("Gender", ["F", "M"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
scholarship_label = st.selectbox("Scholarship", ["No", "Yes"])
hypertension_label = st.selectbox("Hypertension", ["No", "Yes"])
diabetes_label = st.selectbox("Diabetes", ["No", "Yes"])
alcoholism_label = st.selectbox("Alcoholism", ["No", "Yes"])
handcap_label = st.selectbox("Handicap", ["No", "Yes"])
sms_received_label = st.selectbox("SMS Received", ["No", "Yes"])
waiting_days = st.number_input("Waiting Days", min_value=0, max_value=365, value=5)
neighbourhood = st.selectbox("Neighbourhood", neighbourhoods, index=0)

scholarship = yes_no_to_int(scholarship_label)
hypertension = yes_no_to_int(hypertension_label)
diabetes = yes_no_to_int(diabetes_label)
alcoholism = yes_no_to_int(alcoholism_label)
handcap = yes_no_to_int(handcap_label)
sms_received = yes_no_to_int(sms_received_label)

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

    st.write(f"👩‍⚕️ Show probability: {1-prob:.2f}")
    st.write(f"❌ No-show probability: {prob:.2f}")
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