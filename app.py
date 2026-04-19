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

import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# Load .env file for local development
load_dotenv()

# Safely get Groq API key from Streamlit secrets or local .env
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ GROQ_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")

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

# --- LangGraph Definition ---
class AgentState(TypedDict):
    patient_data: dict
    risk_level: str
    probability: float
    guidelines: str
    action_plan: str
    reasoning: str
    final_report: dict

def retrieve_guidelines(state: AgentState):
    guidelines_dict = {
        "High": "Patients with high no-show risk should receive personal phone calls and flexible rescheduling options.",
        "Medium": "Send targeted SMS reminders 24 hours before the appointment.",
        "Low": "Standard scheduling is sufficient. Send standard automated SMS."
    }
    return {"guidelines": guidelines_dict.get(state["risk_level"], "")}

def clinical_reasoning(state: AgentState):
    llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant", api_key=api_key)
    prompt = f"""
    You are an AI Clinical Operations Assistant.
    Given this patient's no-show risk profile, characteristics, and our guidelines, what action plan do you recommend?
    
    Patient Data: {state['patient_data']}
    Risk Level: {state['risk_level']} (Probability: {state['probability']:.2f})
    Standard Guidelines: {state['guidelines']}
    
    Provide a concise action plan (1 sentence) and a brief reason (1 sentence) separated by a newline.
    Format your response EXACTLY like this:
    ACTION: <your action>
    REASON: <your reason>
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        lines = content.strip().split('\n')
        action = [line.replace("ACTION:", "").strip() for line in lines if "ACTION:" in line][0]
        reason = [line.replace("REASON:", "").strip() for line in lines if "REASON:" in line][0]
    except Exception as e:
        action = "Manual clinician review required"
        reason = "LLM structuring failed or API error"
        
    return {"action_plan": action, "reasoning": reason}

def generate_report(state: AgentState):
    report = {
        "risk_level": state["risk_level"],
        "probability": float(state["probability"]),
        "recommended_action": state["action_plan"],
        "reason": state["reasoning"],
        "guideline": state["guidelines"],
        "disclaimer": "AI-generated recommendation via LangGraph & Groq API. Final decision should be made by healthcare professionals."
    }
    return {"final_report": report}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_guidelines", retrieve_guidelines)
workflow.add_node("clinical_reasoning", clinical_reasoning)
workflow.add_node("generate_report", generate_report)

workflow.set_entry_point("retrieve_guidelines")
workflow.add_edge("retrieve_guidelines", "clinical_reasoning")
workflow.add_edge("clinical_reasoning", "generate_report")
workflow.add_edge("generate_report", END)

app_graph = workflow.compile()
# ----------------------------

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
    
    patient_dict = input_df.to_dict('records')[0]
    initial_state = {
        "patient_data": patient_dict,
        "risk_level": risk,
        "probability": prob
    }
    
    # Run the LangGraph
    with st.spinner("AI Agent is reasoning about this patient..."):
        try:
            final_state = app_graph.invoke(initial_state)
            report = final_state["final_report"]
        except Exception as e:
            st.error(f"Error running LangGraph: {str(e)}")
            st.stop()

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