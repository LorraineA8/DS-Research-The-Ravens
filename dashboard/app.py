# app.py – DEPLOY THIS NOW
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Team Ravens T2DM Tool", layout="centered")
st.title("Team Ravens – T2DM Risk Screener")
st.markdown("### Village Health Team Tool • Uganda MoH")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 21, 100, 35)
    bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.1)
    glucose = st.slider("Glucose (mg/dL)", 70, 200, 110)
    preg = st.slider("Pregnancies", 0, 17, 2)
with col2:
    bp = st.slider("Blood Pressure", 60, 140, 90)
    insulin = st.slider("Insulin", 0, 900, 100)
    dpf = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01)
    gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Calculate Risk", type="primary"):
    # Simple rule (will replace with XGBoost tomorrow)
    risk_score = (glucose>126)*40 + (bmi>30)*30 + (age>45)*20 + (preg>6)*10
    risk = "HIGH RISK" if risk_score > 60 else "MEDIUM RISK" if risk_score > 30 else "LOW RISK"
    
    if risk == "HIGH RISK":
        st.error(f"**{risk}** – Refer to health facility TODAY")
    elif risk == "MEDIUM RISK":
        st.warning(f"**{risk}** – Lifestyle advice + recheck in 3 months")
    else:
        st.success(f"**{risk}** – Continue healthy habits")
    
    st.balloons()
