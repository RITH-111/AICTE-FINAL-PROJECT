import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and model columns
model = joblib.load(r"C:\Users\91790\App\water_quality_model.joblib")
scaler = joblib.load(r"C:\Users\91790\App\scaler.pkl")
model_cols = joblib.load(r"C:\Users\91790\App\model_columns.pkl")

# Pollutants to predict
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Extract station IDs from model columns
station_cols = [col for col in model_cols if col.startswith('id_')]
station_ids = ['base'] + [col.split('id_')[1] for col in station_cols]

# Streamlit UI
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")
st.title("💧 Water Pollutants Predictor")
st.markdown("Predict O₂, NO₃, NO₂, SO₄, PO₄, and Cl levels based on **Year**, **Month**, and **Station ID**.")

# Inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
month_input = st.number_input("Enter Month", min_value=1, max_value=12, value=6)
station_id = st.selectbox("Select Station ID", station_ids)

# Predict button
if st.button("🔮 Predict"):
    try:
        # Construct input DataFrame
        input_df = pd.DataFrame({'year': [year_input], 'month': [month_input], 'id': [station_id]})

        # One-hot encode 'id' with drop_first=True (like in training)
        input_encoded = pd.get_dummies(input_df, columns=['id'], drop_first=True)

        # Add any missing columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Align columns
        input_encoded = input_encoded[model_cols]

        # Scale
        input_scaled = scaler.transform(input_encoded)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Show results
        st.subheader(f"📊 Predicted Pollutant Levels for Station ID '{station_id}' in {month_input}/{year_input}:")
        for pol, val in zip(pollutants, prediction):
            st.write(f"**{pol}**: {val:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed:\n\n{e}")
