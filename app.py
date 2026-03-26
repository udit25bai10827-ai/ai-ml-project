import streamlit as st
import pickle

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Pass/Fail Predictor")

st.write("Enter student details:")

# Inputs
study_hours = st.number_input("Study Hours", 0, 12)
attendance = st.number_input("Attendance (%)", 0, 100)
previous_marks = st.number_input("Previous Marks", 0, 100)

# Prediction
if st.button("Predict Result"):
    prediction = model.predict([[study_hours, attendance, previous_marks]])
    st.success(f"Result: {prediction[0]}")