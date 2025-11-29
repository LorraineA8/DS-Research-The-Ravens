"""
Main Streamlit application for Diabetes Clinical Decision Support System (CDSS).

This is the entry point for the Streamlit application.
Streamlit automatically detects pages in the 'pages' directory.
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Diabetes CDSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Theme CSS
def load_theme_css():
    """Load professional theme CSS"""
    theme_css = """
    <style>
    /* Professional Healthcare Theme - Exotic & High-Impact Color Scheme */
    
    /* Color Variables */
    :root {
        --primary-color: #1a4d7a;
        --secondary-color: #2c7a7b;
        --accent-color: #d97757;
        --success-color: #2d8659;
        --warning-color: #d4a574;
        --danger-color: #c85d5d;
        --info-color: #4a90a4;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --light-bg: #f8fafc;
        --border-color: #e2e8f0;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.25rem;
        color: var(--text-secondary);
        text-align: center;
        padding: 0.5rem 0;
        font-weight: 400;
    }
    
    /* Logo Container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.5rem 0 1rem 0;
    }
    
    /* Headers */
    h1 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background-color: #d1fae5 !important;
        border-left: 4px solid var(--success-color) !important;
        color: var(--text-primary) !important;
        border-radius: 4px !important;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        border-left: 4px solid var(--info-color) !important;
        color: var(--text-primary) !important;
        border-radius: 4px !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        border-left: 4px solid var(--warning-color) !important;
        color: var(--text-primary) !important;
        border-radius: 4px !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        border-left: 4px solid var(--danger-color) !important;
        color: var(--text-primary) !important;
        border-radius: 4px !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(26, 77, 122, 0.1) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: var(--text-primary) !important;
        line-height: 1.7 !important;
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-color) !important;
        margin: 2rem 0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--light-bg) !important;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """
    return theme_css

# Apply theme
st.markdown(load_theme_css(), unsafe_allow_html=True)

# Display MoH Logo on left and right of title
import os
logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MoH-Logo.png')

# Create layout with logo on both sides of title
if os.path.exists(logo_path):
    # Three columns: logo left, title center, logo right
    col_logo_left, col_title, col_logo_right = st.columns([1, 3, 1])
    
    with col_logo_left:
        st.image(logo_path, width=120)
    
    with col_title:
        st.markdown('<h1 class="main-header">Diabetes Clinical Decision Support System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Type 2 Diabetes Risk Prediction Tool for Village Health Workers</p>', unsafe_allow_html=True)
    
    with col_logo_right:
        st.image(logo_path, width=120)
else:
    # Fallback if logo not found - just show title
    st.markdown('<h1 class="main-header">Diabetes Clinical Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Type 2 Diabetes Risk Prediction Tool for Village Health Workers</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
    ## Welcome to the Diabetes CDSS
    
    This Clinical Decision Support System helps healthcare workers assess the risk of Type 2 Diabetes 
    in patients using easy-to-measure clinical variables.
    
    ### Features:
    - **Risk Assessment**: Predict diabetes risk based on clinical measurements
    - **Visual Dashboard**: Interactive visualizations of risk factors
    - **Explainable AI**: Understand why the model makes specific predictions
    - **Personalized Recommendations**: Get tailored lifestyle and treatment advice
    
    ### How to Use:
    1. **Login** - Start by logging into the system
    2. **Input Form** - Enter patient clinical measurements
    3. **Risk Dashboard** - View comprehensive risk analysis
    4. **SHAP Explanation** - Understand the prediction reasoning
    
    ---
    
    **Note**: This tool is designed for village health settings where advanced laboratory 
    equipment may not be available. All measurements can be taken with basic medical equipment.
""")

# Sidebar information
with st.sidebar:
    # Use MoH logo in sidebar if available
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    st.markdown("### Navigation")
    st.markdown("""
    Use the sidebar to navigate between pages:
    - **Login**: Authentication
    - **Input Form**: Enter patient data
    - **Risk Dashboard**: View results
    - **SHAP Explanation**: Understand predictions
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This CDSS was developed using:
    - Machine Learning models
    - SHAP for explainability
    - Streamlit for user interface
    
    **Version**: 1.0
    """)

