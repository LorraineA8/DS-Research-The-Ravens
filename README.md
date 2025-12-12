# Diabetes Clinical Decision Support System (CDSS)

A comprehensive Clinical Decision Support System for Type 2 Diabetes risk prediction, designed for village health settings where advanced laboratory equipment may not be available.

##  Tech Stack
<p align="left"> <!-- Languages --> <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" /> <br/> <!-- Frameworks & Libraries --> <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/scikit--learn-ML-FF9A00?style=for-the-badge&logo=scikitlearn&logoColor=white" /> <img src="https://img.shields.io/badge/XGBoost-Model-EB4335?style=for-the-badge&logo=google&logoColor=white" /> <img src="https://img.shields.io/badge/LightGBM-Model-00C800?style=for-the-badge&logo=leaflet&logoColor=white" /> <img src="https://img.shields.io/badge/SHAP-Explainable_AI-8A2BE2?style=for-the-badge&logo=google&logoColor=white" /> <br/> <!-- Data & Visualization --> <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white" /> <img src="https://img.shields.io/badge/Matplotlib-Visualization-003B57?style=for-the-badge&logo=plotly&logoColor=white" /> <br/> <!-- Tools & Workflow --> <img src="https://img.shields.io/badge/Jupyter_Notebook-Analysis-F37626?style=for-the-badge&logo=jupyter&logoColor=white" /> <img src="https://img.shields.io/badge/venv-Virtual_Env-000000?style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Joblib-Model_Serialization-2E5E8A?style=for-the-badge" /> <img src="https://img.shields.io/badge/JSON/Pickle-Metadata_Storage-4A4A4A?style=for-the-badge" /> </p>

##  Project Overview

This project implements a machine learning-based CDSS that:
- Predicts Type 2 Diabetes risk using easy-to-measure clinical variables
- Provides explainable AI insights using SHAP values
- Offers personalized recommendations for lifestyle changes and treatment
- Designed specifically for village health workers with limited resources

##  Project Structure

```
diabetes_cdss_project/
│
├── data/
│   ├── health_care_diabetes.csv                # Original dataset
│   └── cleaned_diabetes_dataset.csv            # Cleaned dataset (generated)
│
├── notebooks/
│   ├── 1_data_understanding_and_cleaning.ipynb      # Data exploration & feature selection
│   ├── 2_model_training_and_evaluation.ipynb        # Model training (5 models)
│   ├── 3_xai_and_fairness.ipynb                     # SHAP explainability
│   └── 4_streamlit_integration_test.ipynb           # Integration testing
│
├── models/
│   ├── best_model_pipeline.pkl                 # Trained best model
│   ├── model_metadata.json                     # Model information
│   ├── selected_features.json                  # Selected features
│   ├── shap_explainer.pkl                      # SHAP explainer
│   └── shap_feature_importance.json            # Feature importance
│
├── streamlit_app/
│   ├── app.py                                  # Main Streamlit app
│   ├── utils.py                                # Utility functions
│   ├── requirements.txt                        # Python dependencies
│   └── pages/
│       ├── 1_Login.py                          # Login page
│       ├── 2_Input_Form.py                     # Patient data input
│       ├── 3_Risk_Dashboard.py                 # Risk visualization
│       └── 4_SHAP_Explain.py                  # SHAP explanations
│
└── README.md                                   # This file
```

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd diabetes_cdss_project
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

##  Running the Notebooks

Execute the notebooks in order to train models and generate required files:

1. **Notebook 1: Data Understanding & Cleaning**
   - Loads and explores the dataset
   - Handles missing values
   - Performs feature selection using domain knowledge, statistics, and ML
   - Saves cleaned dataset

2. **Notebook 2: Model Training & Evaluation**
   - Trains 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
   - Evaluates models using multiple metrics
   - Selects best model based on ROC AUC
   - Saves best model using joblib

3. **Notebook 3: Explainable AI (SHAP)**
   - Creates SHAP explainer
   - Analyzes feature importance
   - Generates explanations for predictions
   - Saves explainer for Streamlit use

4. **Notebook 4: Streamlit Integration Test**
   - Tests model loading
   - Verifies predictions work
   - Checks all required files exist

##  Running the Streamlit Application

1. **Ensure all notebooks have been run** (to generate model files)

2. **Navigate to the streamlit_app directory**
   ```bash
   cd streamlit_app
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

##  Using the Application

### 1. Login Page
- Use demo credentials:
  - Username: `admin` / Password: `admin123`
  - Username: `healthworker` / Password: `health123`
  - Username: `demo` / Password: `demo123`

### 2. Input Form
- Enter patient clinical measurements:
  - Age (years)
  - BMI (Body Mass Index)
  - Blood Pressure (diastolic, mm Hg)
  - Fasting Glucose (mg/dL)
  - Diabetes Pedigree Function (family history risk)
  - Additional optional information
- Click "Predict Diabetes Risk"

### 3. Risk Dashboard
- View comprehensive risk analysis:
  - Risk score gauge chart
  - Age bracket comparison
  - Feature contributions
  - Personalized recommendations
  - Next steps based on risk level

### 4. SHAP Explanation
- Understand why the model made the prediction:
  - Feature contribution charts
  - Natural language explanations
  - Top contributing factors
  - Detailed SHAP values

##  Feature Selection Methodology

Our feature selection uses three complementary approaches:

### 1. Domain Knowledge
Prioritizes features that are:
- Easy to measure in village settings (no advanced equipment)
- Reliable and consistent
- Clinically relevant for Type 2 Diabetes

**Selected Features:**
- Age (very easy - self-reported)
- BMI (easy - scale and measuring tape)
- Blood Pressure (easy - standard BP cuff)
- Glucose (moderate - portable glucose meter)
- Diabetes Pedigree Function (easy - family history questionnaire)

### 2. Statistical Methods
- Correlation analysis with target variable
- Chi-square tests for categorical relationships
- Features with significant statistical association

### 3. Machine Learning
- Random Forest feature importance
- Features with high predictive power
- Cross-validation performance

##  Models Trained

The system trains and compares 5 different models:

1. **Logistic Regression** - Linear model, interpretable
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting, high performance
4. **LightGBM** - Fast gradient boosting
5. **Support Vector Machine (SVM)** - Kernel-based classifier

The best model (based on ROC AUC) is selected and saved for deployment.

##  Model Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (ability to distinguish classes)

##  Explainable AI (XAI)

The system uses SHAP (SHapley Additive exPlanations) to provide:
- **Global Feature Importance**: Which features matter most overall
- **Individual Explanations**: Why a specific prediction was made
- **Feature Contributions**: How each feature affects the prediction
- **Natural Language Explanations**: Easy-to-understand summaries

##  Important Notes

### For Healthcare Workers:
- This tool is for **screening and educational purposes only**
- It is **not a substitute** for professional medical advice
- Always consult qualified healthcare providers for medical decisions
- Use clinical judgment alongside model predictions

### For Developers:
- Ensure all notebooks are run before deploying Streamlit app
- Model files must be in the `models/` directory
- Check that all dependencies are installed
- Test predictions before using in production

##  Troubleshooting

### Model files not found
- **Solution**: Run all notebooks in order (1, 2, 3, 4)

### Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### SHAP explainer errors
- **Solution**: Run Notebook 3 to generate the SHAP explainer

### Streamlit not starting
- **Solution**: Check that you're in the `streamlit_app` directory and run `streamlit run app.py`

##  Dataset Information

- **Dataset**: Pima Augmented Type 2 Diabetes Uganda Dataset
- **Target**: Type 2 Diabetes (Binary: 0 = No Diabetes, 1 = Diabetes)
- **Features**: Clinical measurements (Age, BMI, Glucose, Blood Pressure, etc.)
- **Use Case**: Village health tool - all measurements are easy to obtain

##  Team Information

**Team Ravens** - Data Science Research Project

##  License

This project is for educational and research purposes.

##  Acknowledgments

- Dataset: Pima Indians Diabetes Database (augmented)
- Libraries: scikit-learn, XGBoost, LightGBM, SHAP, Streamlit
- Clinical guidance from healthcare professionals

##  Contact

For questions or issues, please contact the project team.
Rugogamu Noela: rugogamunoela@gmail.com
Lorraine Arinaitwe Paula: lorrainearinaitwe@gmail.com
Ssendi Malon Aloysious: ssendimalon2013@gmail.com

---

**Version**: 1.0  
**Last Updated**: 2025 
**Status**: Active Development

