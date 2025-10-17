import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

with open("credit_card.pkl", "rb") as f:
    credit_card_objects = pickle.load(f)

model = credit_card_objects["model"]
scaler = credit_card_objects["scaler"]
power = credit_card_objects["power"]
log_features = credit_card_objects["log_features"]
numeric_cols = credit_card_objects["numeric_cols"]

st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")
st.title("Credit Card Default Prediction App")

st.write("Predict whether a credit card client will **default next month** based on their account details.")

def user_input_features():
    LIMIT_BAL = st.number_input("Limit Balance", min_value=0, value=20000)
    SEX_option = st.selectbox("Sex", ["Male (1)", "Female (2)"])
    SEX = 1 if "Male" in SEX_option else 2
    EDU_option = st.selectbox("Education", ["Graduate School (1)", "University (2)", "High School (3)", "Others (4)"])
    EDUCATION = int(EDU_option.split("(")[-1][0])
    MAR_option = st.selectbox("Marriage", ["Married (1)", "Single (2)", "Others (3)"])
    MARRIAGE = int(MAR_option.split("(")[-1][0])
    
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

    PAY_0 = st.slider("PAY_0", -2, 8, 0)
    PAY_2 = st.slider("PAY_2", -2, 8, 0)
    PAY_3 = st.slider("PAY_3", -2, 8, 0)
    PAY_4 = st.slider("PAY_4", -2, 8, 0)
    PAY_5 = st.slider("PAY_5", -2, 8, 0)
    PAY_6 = st.slider("PAY_6", -2, 8, 0)

    BILL_AMT1 = st.number_input("Bill Amount 1", value=0)
    BILL_AMT2 = st.number_input("Bill Amount 2", value=0)
    BILL_AMT3 = st.number_input("Bill Amount 3", value=0)
    BILL_AMT4 = st.number_input("Bill Amount 4", value=0)
    BILL_AMT5 = st.number_input("Bill Amount 5", value=0)
    BILL_AMT6 = st.number_input("Bill Amount 6", value=0)

    PAY_AMT1 = st.number_input("Payment Amount 1", value=0)
    PAY_AMT2 = st.number_input("Payment Amount 2", value=0)
    PAY_AMT3 = st.number_input("Payment Amount 3", value=0)
    PAY_AMT4 = st.number_input("Payment Amount 4", value=0)
    PAY_AMT5 = st.number_input("Payment Amount 5", value=0)
    PAY_AMT6 = st.number_input("Payment Amount 6", value=0)

    data = {
        'LIMIT_BAL': LIMIT_BAL, 'SEX': SEX, 'EDUCATION': EDUCATION, 'MARRIAGE': MARRIAGE, 'AGE': AGE,
        'PAY_0': PAY_0, 'PAY_2': PAY_2, 'PAY_3': PAY_3, 'PAY_4': PAY_4, 'PAY_5': PAY_5, 'PAY_6': PAY_6,
        'BILL_AMT1': BILL_AMT1, 'BILL_AMT2': BILL_AMT2, 'BILL_AMT3': BILL_AMT3,
        'BILL_AMT4': BILL_AMT4, 'BILL_AMT5': BILL_AMT5, 'BILL_AMT6': BILL_AMT6,
        'PAY_AMT1': PAY_AMT1, 'PAY_AMT2': PAY_AMT2, 'PAY_AMT3': PAY_AMT3,
        'PAY_AMT4': PAY_AMT4, 'PAY_AMT5': PAY_AMT5, 'PAY_AMT6': PAY_AMT6
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# --- Predict Button ---
if st.button("🔮 Predict Default"):
    # --- Preprocessing ---
    # Apply log transform
    for col in log_features:
        min_val = input_df[col].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            input_df[col] = np.log1p(input_df[col] + shift)
        else:
            input_df[col] = np.log1p(input_df[col])

    # Power Transform only numeric cols
    input_df[numeric_cols] = power.transform(input_df[numeric_cols])

    # Scale features
    scaled_input = scaler.transform(input_df)

    # --- Prediction ---
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[:, 1]

    # --- Display Results ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("Default")
    else:
        st.success("No Default")

