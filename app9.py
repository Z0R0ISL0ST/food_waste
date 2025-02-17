import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load model (assuming you saved it using joblib or pickle)
import joblib
model = joblib.load('crop_yield_model.pkl')

# Streamlit app
st.title('Sustainable Agriculture and Food Waste Detection')

# Input fields
soil_ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=7.0)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=60.0)
fertilizer_amount = st.number_input('Fertilizer Amount (kg)', min_value=0.0, value=100.0)

# Predict button
if st.button('Predict Crop Yield'):
    input_data = pd.DataFrame({
        'soil_ph': [soil_ph],
        'temperature': [temperature],
        'humidity': [humidity],
        'fertilizer_amount': [fertilizer_amount]
    })
    prediction = model.predict(input_data)
    st.write(f'Predicted Crop Yield: {prediction[0]:.2f} units')

# Add more sections for food waste detection if needed