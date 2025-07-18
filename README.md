# 🏦 Loan Approval Prediction App

This is a Machine Learning web application built with **Streamlit** that predicts whether a loan will be approved or not based on applicant details such as income, assets, employment status, and **CIBIL score**. The model uses **Logistic Regression** trained on a real-world-like dataset.

---

## 🔍 Project Overview

Financial institutions need a reliable system to predict loan approvals based on applicant profiles. This project simplifies that process using machine learning. The model is trained on historical data and predicts the likelihood of loan approval.

---

## 📊 Features

- Logistic Regression model trained on structured tabular data
- Interactive web interface using **Streamlit**
- Input features include:
  - Education level
  - Employment status
  - Annual income
  - Loan amount and term
  - Residential, commercial, luxury, and bank asset values
  - CIBIL score (300 to 900)
- Real-time prediction output: ✅ Approved or ❌ Not Approved

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data loading & preprocessing
- **Scikit-learn** – model training and prediction
- **Streamlit** – web app development

---

## 📁 Dataset

The dataset contains the following columns:
- `education` (Graduate / Not Graduate)
- `self_employed` (Yes / No)
- `income_annum` (Annual income)
- `loan_amount`
- `loan_term` (in months)
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `cibil_score` (300 to 900)
- `loan_status` (Approved / Rejected)

> The dataset is stored as: `loan_approval_dataset.csv`

## Author

JayKumar Pravinbhai Mistry
