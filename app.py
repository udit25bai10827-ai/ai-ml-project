import streamlit as st
import pickle

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 AI Student Performance Analyzer")
st.write("Predict student performance using Machine Learning")

# Inputs
study_hours = st.number_input("Study Hours", 0, 12)
attendance = st.number_input("Attendance (%)", 0, 100)
previous_marks = st.number_input("Previous Marks", 0, 100)

st.write("Example: Study Hours = 5, Attendance = 75, Marks = 55")

# Prediction
if st.button("Predict Result"):
    
    # Prediction
    prediction = model.predict([[study_hours, attendance, previous_marks]])
    
    # Confidence score
    proba = model.predict_proba([[study_hours, attendance, previous_marks]])
    confidence = max(proba[0]) * 100

    # Output
    st.success(f"Result: {prediction[0]}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Suggestion
    if prediction[0] == "Fail":
        st.warning("Suggestion: Increase study hours and improve attendance.")
    else:
        st.success("Good performance! Keep it up.")
