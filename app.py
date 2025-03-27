import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("salary_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("💰 Salary Prediction App")
st.write("Enter your Age and Years of Experience to predict your salary.")

# Input fields
experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=18, step=1)

# Predict button
if st.button("Predict Salary"):
    # Prepare input for model
    input_features = np.array([[experience, age]])
    
    # Scale the features
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)[0]

    # Show result
    st.success(f"Predicted Salary: ${prediction:,.2f}")

# Reset Button
if st.button("Reset Inputs"):
    st.experimental_rerun()