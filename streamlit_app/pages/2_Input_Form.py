"""
Input Form Page for Diabetes CDSS

This page allows users to enter patient clinical measurements
for diabetes risk prediction.
"""

import streamlit as st
import pandas as pd
from utils import load_model, prepare_input_data, make_prediction

# Page configuration
st.set_page_config(
    page_title="Input Form - Diabetes CDSS",
    layout="wide"
)

# Professional Theme CSS
def load_theme_css():
    """Load professional theme CSS"""
    return """
    <style>
    :root {
        --primary-color: #1a4d7a;
        --secondary-color: #2c7a7b;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --success-color: #2d8659;
        --danger-color: #c85d5d;
        --warning-color: #d4a574;
        --info-color: #4a90a4;
    }
    
    h1 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
    }
    
    .stForm {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        background-color: #ffffff !important;
    }
    </style>
    """
    
st.markdown(load_theme_css(), unsafe_allow_html=True)

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first to access this page.")
    st.stop()

# Title
st.title("Patient Data Input Form")
st.markdown("Enter the patient's clinical measurements below. All fields marked with * are required.")
st.info("**Important**: No prediction will be made until you click the 'Predict Diabetes Risk' button at the bottom of the form.")

# Load model and features (cache to avoid reloading)
@st.cache_resource
def get_model_info():
    """Load model and return model, metadata, and features."""
    return load_model()

try:
    model, model_metadata, selected_features = get_model_info()
except FileNotFoundError as e:
    st.error(f"Model files not found!")
    st.error(str(e))
    st.markdown("---")
    st.markdown("### How to Fix This")
    st.markdown("""
    You need to run the Jupyter notebooks first to generate the model files:
    
    1. **Notebook 1**: `notebooks/1_data_understanding_and_cleaning.ipynb`
       - Cleans data and selects features
       - Generates: `models/selected_features.json`
    
    2. **Notebook 2**: `notebooks/2_model_training_and_evaluation.ipynb`
       - Trains 5 models and selects the best one
       - Generates: `models/best_model_pipeline.pkl` and `models/model_metadata.json`
    
    3. **Notebook 3**: `notebooks/3_xai_and_fairness.ipynb`
       - Creates SHAP explainer
       - Generates: `models/shap_explainer.pkl`
    
    **Quick Steps:**
    1. Open each notebook in Jupyter/VS Code
    2. Run all cells (Cell → Run All)
    3. Wait for all cells to complete
    4. Then return here and refresh the page
    """)
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.exception(e)
    st.stop()

# Create form
with st.form("patient_input_form"):
    st.markdown("### Patient Information")
    
    # Gender selection - must be first
    gender = st.selectbox(
        "Patient Gender *",
        ["Male", "Female"],
        index=0,
        help="Select the patient's gender"
    )
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        
        # Age - Number input with validation
        age = st.number_input(
            "Age (years) *",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            help="Enter the patient's age (must be 18 or older)"
        )
        
        # Show age validation error if age < 18
        if age < 18:
            st.error("❌ Error: Age must be 18 years or older. Please enter a valid age.")
        
        # BMI - Number input
        bmi = st.number_input(
            "BMI (Body Mass Index) *",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="BMI = weight (kg) / height (m)². Normal range: 18.5-24.9"
        )
        
        # Blood Pressure - Number input
        blood_pressure = st.number_input(
            "Blood Pressure (Diastolic, mm Hg) *",
            min_value=40.0,
            max_value=150.0,
            value=70.0,
            step=1.0,
            help="Diastolic blood pressure reading"
        )
    
    with col2:
        st.markdown("#### Clinical Measurements")
        
        # Glucose - Number input
        glucose = st.number_input(
            "Fasting Glucose (mg/dL) *",
            min_value=50.0,
            max_value=300.0,
            value=100.0,
            step=1.0,
            help="Fasting blood glucose level. Normal: <100, Pre-diabetic: 100-125, Diabetic: ≥126"
        )
        
        # Diabetes Pedigree Function - Number input
        # This represents family history risk
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function *",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Genetic risk score based on family history (0-2.5)"
        )
    
    # Pregnancies field - only show for Female (always show if Female, regardless of selected_features)
    st.markdown("---")
    if gender == "Female":
        # Show pregnancies dropdown for Female patients
        pregnancies = st.selectbox(
            "How many pregnancies? *",
            list(range(0, 21)),  # Options from 0 to 20
            index=0,
            help="Number of times pregnant (for female patients only)"
        )
    else:
        # For Male patients, set pregnancies to 0
        pregnancies = 0
    
    # Additional optional features
    st.markdown("---")
    st.markdown("### Additional Information (Optional)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        physical_activity = st.selectbox(
            "Physical Activity Level",
            ["Low", "Moderate", "High"],
            index=1,
            help="Patient's regular physical activity level"
        )
        
        diet_quality = st.selectbox(
            "Diet Quality",
            ["Poor", "Fair", "Good", "Excellent"],
            index=2,
            help="Overall diet quality assessment"
        )
    
    with col4:
        alcohol_use = st.selectbox(
            "Alcohol Use",
            ["None", "Occasional", "Regular", "Heavy"],
            index=0,
            help="Alcohol consumption frequency"
        )
        
        smoking = st.selectbox(
            "Smoking Status",
            ["Non-smoker", "Former smoker", "Current smoker"],
            index=0,
            help="Current smoking status"
        )
    
    # Family history
    family_history = st.radio(
        "Family History of Diabetes",
        ["No", "Yes"],
        index=0,
        horizontal=True,
        help="Does the patient have a family history of diabetes?"
    )
    
    # Submit button
    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True,
        type="primary"
    )
    
    # CRITICAL: Only make prediction when button is explicitly clicked
    # This ensures no predictions happen automatically
    if submitted:
        # Validate age on submission
        if age < 18:
            st.error("❌ Error: Age must be 18 years or older. Please enter a valid age before submitting.")
            st.stop()
        
        # Validate gender (should not be needed, but just in case)
        if gender not in ["Male", "Female"]:
            st.error("❌ Error: Please select a valid gender.")
            st.stop()
        
        # Prepare input data
        user_input = {
            'Age': age,
            'BMI': bmi,
            'BloodPressure': blood_pressure,
            'Glucose': glucose,
            'DiabetesPedigreeFunction': diabetes_pedigree
        }
        
        # Add pregnancies - always include if it's in selected_features
        if 'Pregnancies' in selected_features:
            if gender == "Female":
                user_input['Pregnancies'] = pregnancies
            else:
                # For Male patients, set pregnancies to 0
                user_input['Pregnancies'] = 0
        
        # Add any other required features with default values
        for feature in selected_features:
            if feature not in user_input:
                user_input[feature] = 0
        
        # Prepare data for model
        input_data = prepare_input_data(user_input, selected_features)
        
        # Make prediction
        try:
            prediction_result = make_prediction(model, input_data)
            
            # Store results in session state for use in dashboard
            st.session_state.prediction_result = prediction_result
            st.session_state.user_input = user_input
            st.session_state.input_data = input_data
            
            # Show quick result
            risk_score = prediction_result['risk_score']
            risk_level = prediction_result['risk_level']
            
            st.success("Prediction completed!")
            
            # Display risk level with color coding
            if risk_level == "High":
                st.error(f"**High Risk Detected**: {risk_score:.1f}% risk of Type 2 Diabetes")
            elif risk_level == "Medium":
                st.warning(f"**Moderate Risk**: {risk_score:.1f}% risk of Type 2 Diabetes")
            else:
                st.info(f"**Low Risk**: {risk_score:.1f}% risk of Type 2 Diabetes")
            
            st.info("Navigate to the **Risk Dashboard** page to see detailed analysis and recommendations.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

# Information section
st.markdown("---")
st.markdown("### Measurement Guidelines")

with st.expander("How to measure each variable"):
    st.markdown("""
    **Age**: Patient's current age in years.
    
    **BMI (Body Mass Index)**:
    - Measure patient's weight (kg) and height (m)
    - Calculate: BMI = weight ÷ (height)²
    - Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30
    
    **Blood Pressure**:
    - Use a standard blood pressure cuff
    - Record diastolic pressure (bottom number)
    - Normal: <80, Elevated: 80-89, High: ≥90
    
    **Fasting Glucose**:
    - Use a portable glucose meter
    - Patient should fast for at least 8 hours
    - Normal: <100 mg/dL, Pre-diabetic: 100-125, Diabetic: ≥126
    
    **Diabetes Pedigree Function**:
    - Based on family history of diabetes
    - Calculate based on affected relatives
    - Higher value = stronger family history
    """)

