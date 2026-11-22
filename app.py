"""
Telecom Churn Predictor — Streamlit app
XGBoost predicts churn probability; Google Gemini writes a retention report.
Run locally:  streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Churn Predictor AI", page_icon="📡", layout="wide")


# ---------- Load model + metadata (cached so it loads only once) ----------
@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("churn_model.json")          # saved by the notebook
    with open("columns.json") as f:
        columns = json.load(f)                    # column order used in training
    with open("encoders.json") as f:
        encoders = json.load(f)                   # {column: [classes]} -> index = encoded value
    return model, columns, encoders


model, columns, encoders = load_artifacts()


def options(col):
    """Dropdown choices come straight from the training data's categories."""
    return encoders[col]


def encode(user: dict) -> pd.DataFrame:
    """Turn the form inputs into the exact numeric row the model was trained on."""
    row = {}
    for col in columns:
        if col in encoders:
            row[col] = encoders[col].index(user[col])   # text -> number (same mapping as training)
        else:
            row[col] = user[col]                        # already numeric
    return pd.DataFrame([row])[columns]


# ---------- Gemini (optional: app still works without a key) ----------
def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def generate_report(user: dict, prob: float) -> str:
    from google import genai

    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    model_name = get_secret("GEMINI_MODEL", "gemini-3.5-flash")
    level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    profile = "\n".join(f"- {k}: {v}" for k, v in user.items())
    prompt = f"""You are a senior telecom business analyst.
A machine learning model scored this customer's churn probability at {prob*100:.1f}% ({level} risk).

Customer profile:
{profile}

Write a concise report in exactly 3 short sections:
1. Risk analysis - why the customer is at this risk level.
2. Key risk factors - the 2-3 strongest drivers visible in the profile.
3. Retention plan - 3 specific, actionable steps the company should take.
Use plain professional language and only facts from the profile."""
    return client.models.generate_content(model=model_name, contents=prompt).text


# ---------- UI ----------
st.title("📡 Telecom Customer Churn Predictor")
st.caption("XGBoost churn model + Gemini-generated retention report")

with st.sidebar:
    st.header("About")
    st.write("Predicts whether a telecom customer is likely to leave, "
             "trained on the IBM Telco Customer Churn dataset (7,032 customers).")
    st.write("Model: XGBoost · ROC-AUC ≈ 0.83")
    st.caption("Built by Divya Patel")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", options("gender"))
    senior = st.selectbox("Senior citizen", options("SeniorCitizen"))
    partner = st.selectbox("Has partner", options("Partner"))
    dependents = st.selectbox("Has dependents", options("Dependents"))
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with c2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", options("PhoneService"), index=1)
    lines = st.selectbox("Multiple lines", options("MultipleLines"))
    internet = st.selectbox("Internet service", options("InternetService"), index=1)
    security = st.selectbox("Online security", options("OnlineSecurity"))
    backup = st.selectbox("Online backup", options("OnlineBackup"))
    device = st.selectbox("Device protection", options("DeviceProtection"))
with c3:
    st.subheader("Plan & billing")
    tech = st.selectbox("Tech support", options("TechSupport"))
    tv = st.selectbox("Streaming TV", options("StreamingTV"))
    movies = st.selectbox("Streaming movies", options("StreamingMovies"))
    contract = st.selectbox("Contract", options("Contract"))
    paperless = st.selectbox("Paperless billing", options("PaperlessBilling"), index=1)
    payment = st.selectbox("Payment method", options("PaymentMethod"), index=2)
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)

if st.button("Predict churn", type="primary", use_container_width=True):
    user = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2),   # approximation: tenure x monthly charge
    }
    prob = float(model.predict_proba(encode(user))[0][1])

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob*100:.1f}%")
    m2.metric("Risk level", "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")
    st.progress(prob)
    if prob > 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

    st.subheader("🤖 AI retention report")
    if not get_secret("GEMINI_API_KEY"):
        st.info("Add GEMINI_API_KEY (see README) to enable the AI report.")
    else:
        with st.spinner("Gemini is analysing the profile..."):
            try:
                report = generate_report(user, prob)
                st.markdown(report)
                st.download_button("Download report", report, "churn_report.txt")
            except Exception as e:
                st.warning(f"Could not generate the AI report: {e}")
