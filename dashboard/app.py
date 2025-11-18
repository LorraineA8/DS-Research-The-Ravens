import streamlit as st
st.set_page_config(page_title="Team Ravens T2DM", layout="wide")
st.title("Team Ravens – Type 2 Diabetes Risk Predictor (Uganda)")
st.markdown("### Community Health Worker Tool – Based on Pima Augmented Dataset")

c1, c2 = st.columns(2)
with c1:
    preg = st.slider("Pregnancies", 0, 20, 3)
    glu  = st.slider("Glucose (mg/dL)", 0, 200, 120)
    bp   = st.slider("Blood Pressure", 0, 130, 70)
    skin = st.slider("Skin Thickness (mm)", 0, 100, 20)
with c2:
    ins  = st.slider("Insulin", 0, 900, 80)
    bmi  = st.slider("BMI", 0.0, 70.0, 32.0)
    dpf  = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)
    age  = st.slider("Age", 21, 100, 35)

if st.button("Predict Diabetes Risk"):
    st.success("Model will be ready tomorrow!")
    st.balloons()
