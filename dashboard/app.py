# dashboard/app.py  (replace the old one)
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the scaler (will exist tomorrow)
try:
    scaler = joblib.load('models/scaler.pkl')
except:
    scaler = None

st.set_page_config(page_title="Team Ravens T2DM", layout="wide")
st.title("Team Ravens – Type 2 Diabetes Risk Predictor")
st.markdown("#### Uganda Community Health Tool — Day 1 working version")

c1, c2 = st.columns(2)
with c1:
    preg = st.slider("Pregnancies", 0, 20, 3)
    glu  = st.slider("Glucose (mg/dL)", 0, 200, 120)
    bp   = st.slider("Blood Pressure", 0, 130, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
with c2:
    ins  = st.slider("Insulin", 0, 900, 80)
    bmi  = st.slider("BMI", 0.0, 70.0, 32.0)
    dpf  = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)
    age  = st.slider("Age", 21, 100, 35)

if st.button("Predict Risk Now"):
    # Fake model for today (tomorrow we replace with XGBoost)
    risk_score = 0.5 + (glu-120)/200 + (bmi-32)/20 + (age-35)/50
    risk_score = np.clip(risk_score, 0, 1)
    
    if risk_score > 0.7:
        st.error(f"High Risk ({risk_score:.1%}) – Refer immediately")
    elif risk_score > 0.4:
        st.warning(f"Medium Risk ({risk_score:.1%}) – Lifestyle advice")
    else:
        st.success(f"Low Risk ({risk_score:.1%})")
    st.balloons()
