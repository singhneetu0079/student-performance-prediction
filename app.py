
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/student_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Application heading
st.title("🎓 Student Performance Prediction")

st.write(
    "Enter the student's details below to predict "
    "the expected final examination score."
)

# User inputs
study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance percentage",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_score = st.number_input(
    "Previous examination score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

sleep_hours = st.number_input(
    "Sleep hours per day",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

practice_tests = st.number_input(
    "Number of practice tests",
    min_value=0,
    max_value=100,
    value=5
)

# Prediction button
if st.button("Predict Final Score"):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "practice_tests": practice_tests
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"Predicted Final Score: {prediction:.2f}/100"
    )
