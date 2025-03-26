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

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("salary_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar for input
st.sidebar.header("User Input")
experience = st.sidebar.slider("Years of Experience", 0, 20, 3)
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor’s", "Master’s", "PhD"])
skills = st.sidebar.multiselect("Skills", ["Python", "R", "SQL", "Machine Learning", "Deep Learning"])

# Convert education to numerical value
education_map = {"High School": 0, "Bachelor’s": 1, "Master’s": 2, "PhD": 3}
education_num = education_map[education]

# Process input
features = np.array([[experience, education_num]])
features_scaled = scaler.transform(features)

# Predict salary
predicted_salary = model.predict(features_scaled)[0]

# Display results
st.title("💰 Salary Prediction App")
st.subheader("Get an estimate of your salary based on experience and education!")

st.metric("Predicted Salary", f"${predicted_salary:,.2f}")

st.success("Prediction successful! Adjust the inputs on the sidebar to see different results.")

# Add a reset button
if st.button("Reset Inputs"):
    st.experimental_rerun()
