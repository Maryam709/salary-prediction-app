import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

# Streamlit UI
st.title("Salary Prediction App")
st.write("Enter the required values to predict salary.")

# Input fields for 11 features
experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)
feature_2 = st.number_input("Feature 2", min_value=0, step=1)
feature_3 = st.number_input("Feature 3", min_value=0, step=1)
feature_4 = st.number_input("Feature 4", min_value=0, step=1)
feature_5 = st.number_input("Feature 5", min_value=0, step=1)
feature_6 = st.number_input("Feature 6", min_value=0, step=1)
feature_7 = st.number_input("Feature 7", min_value=0, step=1)
feature_8 = st.number_input("Feature 8", min_value=0, step=1)
feature_9 = st.number_input("Feature 9", min_value=0, step=1)
feature_10 = st.number_input("Feature 10", min_value=0, step=1)
feature_11 = st.number_input("Feature 11", min_value=0, step=1)

# Predict button
if st.button("Predict Salary"):
    # Prepare input for model
    input_features = np.array([[experience, feature_2, feature_3, feature_4, feature_5, 
                                 feature_6, feature_7, feature_8, feature_9, feature_10, feature_11]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]

    # Show result
    st.success(f"Predicted Salary: ${prediction:,.2f}")
