import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('crop_yield_data.csv')

# Features and target
X = df[['soil_ph', 'temperature', 'humidity', 'fertilizer_amount']]
y = df['crop_yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Sustainable Agriculture and crop yeild prediction')

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
