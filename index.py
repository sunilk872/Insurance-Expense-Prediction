import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pickle
from io import BytesIO

# Page Configuration
st.set_page_config(page_title="Medical Expense Prediction", page_icon="💰", layout="wide")

# Custom CSS for Background and Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stApp {
        background-color: #eef2f7;
    }
    .stSidebar {
        background-color: #dce7f3;
    }
    .stTable, .dataframe tbody tr, .dataframe thead th {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .prediction-box {
        font-size: 24px;
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("💰 Medical Expense Prediction App")
st.markdown("Predict medical expenses based on user inputs!")

# Sidebar for User Input
st.sidebar.header("🔹 User Input Parameters")
def user_input_features():
    AGE = st.sidebar.slider("Select Age", 0, 100, 25)
    SEX = st.sidebar.radio("Gender", ["Male", "Female"], index=0)
    BMI = st.sidebar.slider("Select BMI", 10.0, 50.0, 25.0, step=0.1)
    CHILDREN = st.sidebar.slider("Number of Children", 0, 5, 0)
    SMOKER = st.sidebar.radio("Smoker?", ["Yes", "No"], index=1)
    REGION = st.sidebar.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"], index=0)
    
    data = {
        "age": AGE,
        "sex": 1 if SEX == "Male" else 0,
        "bmi": BMI,
        "children": CHILDREN,
        "smoker": 1 if SMOKER == "Yes" else 0,
        "region": ["Southwest", "Southeast", "Northwest", "Northeast"].index(REGION),
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Display User Inputs
st.subheader("🔍 User Inputs")
st.dataframe(df.style.set_properties(**{"background-color": "#ffffff", "border-radius": "10px"}))

# Load Model
MODEL_URL = "https://github.com/sunilk872/Insurance-Expense-Prediction/raw/main/finalized_model.pkl"
try:
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        loaded_model = pickle.load(BytesIO(response.content))
    else:
        st.error("❌ Failed to fetch the model from GitHub.")
        loaded_model = None
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    loaded_model = None

# Prediction
if loaded_model:
    prediction = loaded_model.predict(df)
    prediction = np.round(prediction, 2)
    
    # Display Prediction
    st.subheader("📢 Prediction Result")
    st.markdown(f'<div class="prediction-box">✅ Estimated Medical Expenses: ${prediction[0]:,.2f}</div>', unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📊 How Inputs Affect Prediction")
    fig, ax = plt.subplots(figsize=(8, 5))
    input_labels = ["Age", "Gender", "BMI", "Children", "Smoker", "Region"]
    input_values = [
        df["age"].iloc[0],
        int(df["sex"].iloc[0]),
        df["bmi"].iloc[0],
        int(df["children"].iloc[0]),
        int(df["smoker"].iloc[0]),
        int(df["region"].iloc[0]),
    ]
    sns.barplot(x=input_labels, y=input_values, ax=ax, palette="magma")
    ax.set_title("User Input Parameters")
    ax.set_ylabel("Value")
    ax.set_xlabel("Parameters")
    st.pyplot(fig)
    
    # Histogram of Predictions
    st.subheader("📈 Predicted Expenses Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(prediction, kde=True, ax=ax, color="blue", bins=10)
    ax.set_title("Predicted Medical Expenses Distribution")
    ax.set_xlabel("Medical Expenses ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
