"""
Login Page for Diabetes CDSS

This page handles user authentication.
For demo purposes, we use simple username/password authentication.
In production, this should be replaced with secure authentication.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Login - Diabetes CDSS",
    layout="centered"
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
        --light-bg: #f8fafc;
        --border-color: #e2e8f0;
        --success-color: #2d8659;
        --danger-color: #c85d5d;
        --warning-color: #d4a574;
    }
    
    .login-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 2.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }
    
    h1 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
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
    
    .stSuccess {
        background-color: #d1fae5 !important;
        border-left: 4px solid var(--success-color) !important;
        border-radius: 4px !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        border-left: 4px solid var(--danger-color) !important;
        border-radius: 4px !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        border-left: 4px solid var(--warning-color) !important;
        border-radius: 4px !important;
    }
    </style>
    """
    
st.markdown(load_theme_css(), unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Title
st.title("Welcome to the Village Diabetes Clinical Decision Support Tool")

# Check if already authenticated
if st.session_state.authenticated:
    st.success(f"You are logged in as: **{st.session_state.username}**")
    st.info("Navigate to other pages using the sidebar.")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
else:
    # Login form
    st.markdown("### Please enter your credentials")
    
    # For demo purposes, we'll use simple authentication
    # In production, use proper authentication (OAuth, database, etc.)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            # Simple authentication (for demo)
            # In production, validate against a database or authentication service
            if username and password:
                # Demo credentials (replace with actual authentication)
                valid_credentials = {
                    "admin": "admin123",
                    "healthworker": "health123",
                    "demo": "demo123"
                }
                
                if username in valid_credentials and password == valid_credentials[username]:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Login successful! Welcome, {username}.")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
            else:
                st.warning("Please enter both username and password.")
    
    # Demo credentials info
    st.markdown("---")
    st.markdown("### Demo Credentials")
    st.info("""
    For demonstration purposes, you can use:
    - **Username**: `admin` / **Password**: `admin123`
    - **Username**: `healthworker` / **Password**: `health123`
    - **Username**: `demo` / **Password**: `demo123`
    """)
    
    st.markdown("---")
    st.markdown("""
    **Note**: This is a demonstration system. In production, implement proper 
    authentication with secure password hashing and user management.
    """)

