import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load('admission_best_model.joblib')

st.title("🎓 Admission Chance Prediction App")
st.write("Predict your chance of admission to a graduate program.")

# User inputs
gre = st.slider("GRE Score", 290, 340, 320)
toefl = st.slider("TOEFL Score", 92, 120, 105)
rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.slider("Undergraduate CGPA", 6.8, 9.92, 8.5)
research = st.selectbox("Research Experience", [0, 1])

if st.button("Predict Admission Chance"):
    features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    prediction = model.predict(features)[0]
    st.success(f"🎯 Estimated Chance of Admission: {prediction:.2f}")
