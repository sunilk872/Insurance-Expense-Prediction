import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pickle
from io import BytesIO

# Set the page config for a better UI experience
st.set_page_config(
    page_title="Medical Expense Prediction",
    page_icon="💰",
    layout="wide"
)

# Apply custom styles
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        .sidebar .sidebar-content {
            background-color: #e3e6f3;
        }
        .stMarkdown h1 {
            color: #374151;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with emoji for better UX
st.title("💰 Medical Expense Prediction")

# Sidebar header for input parameters
st.sidebar.header("🔢 User Input Parameters")

def user_input_features():
    AGE = st.sidebar.slider("Insert Age", min_value=18, max_value=100, value=30)
    SEX = st.sidebar.radio("Gender:", ("Male", "Female"))
    BMI = st.sidebar.slider("Insert BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    CHILDREN = st.sidebar.selectbox("Number of Children:", [0, 1, 2, 3, 4, 5])
    SMOKER = st.sidebar.radio("Smoker:", ("Yes", "No"))
    REGION = st.sidebar.selectbox("Region:", ["Southwest", "Southeast", "Northwest", "Northeast"])
    
    data = {
        "age": AGE,
        "sex": 1 if SEX == "Male" else 0,
        "bmi": BMI,
        "children": CHILDREN,
        "smoker": 1 if SMOKER == "Yes" else 0,
        "region": ["Southwest", "Southeast", "Northwest", "Northeast"].index(REGION)
    }
    return pd.DataFrame(data, index=[0])

# Collect user inputs
df = user_input_features()

# Display user inputs in an elegant format
st.subheader("📊 User Input Parameters")
st.write(df.style.set_properties(**{'background-color': '#f8f9fa', 'color': '#343a40', 'border-radius': '10px'}))

# Load the trained model from GitHub
MODEL_URL = "https://github.com/sunilk872/Insurance-Expense-Prediction/raw/main/finalized_model.pkl"
loaded_model = None

try:
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        loaded_model = pickle.load(BytesIO(response.content))
    else:
        st.error("🚨 Failed to fetch the model file from GitHub.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Prediction
if loaded_model:
    prediction = loaded_model.predict(df)
    prediction = np.round(prediction, 2)
    
    # Display the prediction in a highlighted box
    st.subheader("💡 Prediction Result")
    st.markdown(
        f"""
        <div style="padding: 15px; background-color: #d1e7dd; border-radius: 10px; font-size: 18px;">
            ✅ **Predicted Medical Expenses:** ${prediction[0]:,.2f}
        </div>
        """, unsafe_allow_html=True
    )
    
    # Visualization Section
    st.subheader("📉 Visualization: How Inputs Affect Prediction")
    
    # Bar chart of user inputs
    fig, ax = plt.subplots(figsize=(8, 5))
    input_labels = ["Age", "Gender", "BMI", "Children", "Smoker", "Region"]
    input_values = [
        df["age"].iloc[0],
        df["sex"].iloc[0],
        df["bmi"].iloc[0],
        df["children"].iloc[0],
        df["smoker"].iloc[0],
        df["region"].iloc[0]
    ]
    
    sns.barplot(x=input_labels, y=input_values, ax=ax, palette="magma")
    ax.set_title("User Input Parameters", fontsize=14)
    ax.set_ylabel("Value")
    ax.set_xlabel("Parameters")
    st.pyplot(fig)
    
    # Histogram for predicted expenses
    st.subheader("📊 Distribution of Predicted Medical Expenses")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(prediction, kde=True, ax=ax, color="blue", bins=10)
    ax.set_title("Predicted Medical Expenses Distribution", fontsize=14)
    ax.set_xlabel("Medical Expenses ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
