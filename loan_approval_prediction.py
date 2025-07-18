import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# Title
st.title("🏦 Loan Approval Prediction APP")

# One-time model training using session_state
if "model" not in st.session_state:
    # Load and clean dataset
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()

    # Encode categorical and target variables
    df["education"] = df["education"].str.strip().replace({"Graduate": 1, "Not Graduate": 0})
    df["self_employed"] = df["self_employed"].str.strip().replace({"Yes": 1, "No": 0})
    df["loan_status"] = df["loan_status"].str.strip().replace({"Approved": 1, "Rejected": 0}).astype(int)

    # Split features and labels
    X = df[["education", "self_employed", "income_annum", "loan_amount", "loan_term", "cibil_score",
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value"]]
    y = df["loan_status"]

    # Train/test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save in session state
    st.session_state.model = model
    st.session_state.columns = X.columns
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.success(f"✅ Model trained once and stored in session (Accuracy: {accuracy:.2%})")

# User Inputs

st.subheader("📋 Enter Applicant Details")
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0, max_value=1000000000, step=1000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, max_value=1000000000, step=1000)
loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=240)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, max_value=100000000, step=1000)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, max_value=100000000, step=1000)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, max_value=100000000, step=1000)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, max_value=100000000, step=1000)

# Prediction
if st.button("🔍 Predict Loan Status"):
    # Convert inputs
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0

    input_data = pd.DataFrame([[
        education_val, self_employed_val, income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]], columns=st.session_state.columns)

    model = st.session_state.model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Result Output
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
    st.markdown(f"**📈 Approval Probability:** `{probability:.2%}`")

    # Gauge Chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Loan Approval Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if prediction == 1 else "red"},
               'steps': [
                   {'range': [0, 50], 'color': "#ffcccc"},
                   {'range': [50, 75], 'color': "#fff9c4"},
                   {'range': [75, 100], 'color': "#c8e6c9"}
               ]}
    ))
    st.plotly_chart(gauge)

    # Bar Chart for Input Features
    st.subheader("📊 Applicant Feature Overview")
    feature_values = input_data.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_values.index, feature_values.values, color='skyblue')
    ax.set_xlabel("Value")
    ax.set_title("Input Feature Values")
    st.pyplot(fig)
