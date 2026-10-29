
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Load saved model
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with open(THRESHOLD_PATH, "r") as file:
    threshold = json.load(file)["threshold"]

# -------------------------
# Dashboard heading
# -------------------------
st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to estimate their likelihood of leaving."
)

st.info(f"Classification threshold: {threshold:.0%}")

# -------------------------
# Customer input form
# -------------------------
with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# -------------------------
# Prediction
# -------------------------
if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer_df = pd.DataFrame([customer])
    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = probability >= threshold

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with result_col2:
        st.metric("Decision Threshold", f"{threshold:.0%}")

    st.progress(probability)

    if prediction:
        st.error(
            "High Risk Classification: This customer is predicted to churn."
        )
    else:
        st.success(
            "Low Risk Classification: This customer is predicted not to churn."
        )

    st.caption(
        "This is a model estimate, not a guarantee of future customer behavior."
    )
