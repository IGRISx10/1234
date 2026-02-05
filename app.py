import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Player Injury Predictor")

model = joblib.load("injury_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏥 Player Injury Percentage Predictor")

st.write("Enter player attributes to predict injury risk")

# Input fields (dynamic)
input_data = {}
sample_df = pd.read_excel("data/Training.xlsx")
feature_columns = sample_df.drop("INJURY PERCENTAGE", axis=1).columns

for col in feature_columns:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Predict Injury Percentage"):
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    st.success(f"🩺 Predicted Injury Percentage: **{prediction:.2f}%**")
