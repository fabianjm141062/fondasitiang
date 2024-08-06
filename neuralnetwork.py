# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Title
st.title("Prediksi Daya Dukung Fondasi Tiang")

# Sidebar inputs
depth = st.sidebar.slider("Kedalaman Tiang (m)", min_value=1.0, max_value=50.0, value=10.0, step=0.1)
diameter = st.sidebar.slider("Diameter Tiang (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
soil_type1 = st.sidebar.selectbox("Jenis Tanah Lapisan 1", options=["Pasir", "Lempung", "Batu"])
soil_type2 = st.sidebar.selectbox("Jenis Tanah Lapisan 2", options=["Pasir", "Lempung", "Batu"])
soil_type3 = st.sidebar.selectbox("Jenis Tanah Lapisan 3", options=["Pasir", "Lempung", "Batu"])

# Preprocess the inputs
input_data = np.array([[depth, diameter, soil_type1, soil_type2, soil_type3]])
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_data.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for predicting bearing capacity
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Load trained model weights (assuming weights are saved after training)
model.load_weights('model_weights.h5')

# Predict the bearing capacity
prediction = model.predict(input_data_scaled)

# Display the prediction
st.write(f"Daya Dukung Fondasi Tiang: {prediction[0][0]:.2f} kN")
