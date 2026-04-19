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

def predict_risk(model, patient_data):
    prob = model.predict_proba(patient_data)[0][1]
    
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    
    return prob, risk

def agent_decision(risk, patient_data):
    lead_time = patient_data["WaitingDays"].values[0]
    past_sms_received = patient_data["SMS_received"].values[0]
    
    if risk == "High":
        action = "Call + SMS Reminder + Reschedule Option"
        reason = "High predicted no-show probability"
    elif risk == "Medium":
        action = "Send SMS Reminder"
        reason = "Moderate risk of missing appointment"
    else:
        action = "No action needed"
        reason = "Low risk"
    
    return {
        "action": action,
        "reason": reason
    }

guidelines = {
    "High": "Patients with high no-show risk should receive phone calls and flexible rescheduling.",
    "Medium": "Send SMS reminders 24 hours before appointment.",
    "Low": "Standard scheduling is sufficient."
}

def generate_report(prob, risk, decision):
    return {
        "risk_level": risk,
        "probability": float(prob),
        "recommended_action": decision["action"],
        "reason": decision["reason"],
        "guideline": guidelines[risk],
        "disclaimer": "AI-generated recommendation. Final decision should be made by healthcare professionals."
    }

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

    prob, risk = predict_risk(model, input_df)
    decision = agent_decision(risk, input_df)
    report = generate_report(prob, risk, decision)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"👩‍⚕️ Show probability: {1-prob:.2%}")
    with col2:
        st.warning(f"❌ No-show probability: {prob:.2%}")

    st.subheader("Prediction Confidence")
    st.write("Show probability")
    st.progress(float(1 - prob))
    st.write("No-show probability")
    st.progress(float(prob))

    st.subheader("🤖 AI Agent Recommendation")
    st.markdown(f"**Risk Level:** `{report['risk_level']}`")
    st.markdown(f"**Recommended Action:** {report['recommended_action']}")
    st.markdown(f"**Reason:** {report['reason']}")
    st.markdown(f"**Guidelines:** {report['guideline']}")
    
    st.caption(f"⚠️ {report['disclaimer']}")
with st.expander("How this works"):
    st.write("""
    The model uses patient demographics and appointment details 
    (e.g., age, waiting days, SMS reminders) to predict the likelihood of a no-show.
    """)