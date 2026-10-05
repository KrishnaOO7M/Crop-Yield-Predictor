import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="UP Crop Yield Predictor", page_icon="🌾", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load('crop_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    rf_model, scaler, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("Error: Model artifacts not found. Please run your training script first to generate the .pkl files.")
    st.stop()


st.title("🌾 Uttar Pradesh Crop Yield Predictor")
st.markdown("""
Welcome to the predictive crop yield tool, developed as part of a grassroots social internship. 
Enter the soil, climate, and crop details below to estimate the yield per hectare.
""")

st.header("1. Environmental & Soil Data")
col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0, step=10.0)
with col2:
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0.0, value=150.0, step=5.0)
with col3:
    pesticide = st.number_input("Pesticide Used (kg/ha)", min_value=0.0, value=2.5, step=0.1)

st.header("2. Crop Details")

seasons = ['Kharif     ', 'Rabi       ', 'Whole Year ', 'Summer     ', 'Autumn     ', 'Winter     ']
crops = ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Potato', 'Mustard', 'Soyabean', 'Onion', 'Cotton(lint)', 'Groundnut', 'Safflower', 'Mesta', 'Castor seed', 'Guar seed']

season = st.selectbox("Select Season", seasons)
crop = st.selectbox("Select Crop", crops)

if st.button("Predict Yield", type="primary"):
    
    input_data = {
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Season': season,
        'Crop': crop
    }
    
    input_df = pd.DataFrame([input_data])
    
    numerical_cols = ['Annual_Rainfall', 'Fertilizer', 'Pesticide']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    input_encoded = pd.get_dummies(input_df, columns=['Season', 'Crop'])
    
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = rf_model.predict(input_encoded)[0]
    
    st.success("Prediction Complete!")
    st.metric(label="Estimated Yield (Tonnes per Hectare)", value=f"{prediction:.2f} t/ha")
    
    st.info("💡 Note: This prediction is based on historical machine learning patterns tailored specifically to Uttar Pradesh.")