import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

# Model loading
model_path = 'C:/Users/N/Desktop/New folder/lstm_model.keras'
model = None
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully.")
except ValueError as e:
    st.error(f"Error loading model: {e}")

# Define the columns for sensors
features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                     's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# Add inputs for each feature
st.title("Manual Sensor Data Input")
st.write("Enter sensor values to predict machine failure probability.")

sensor_values = {}
for feature in features_col_name:
    sensor_values[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Once the user has entered the sensor data, we scale it and make predictions
if st.button("Predict Failure Probability"):
    if model:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame([sensor_values])

        # Scale the input data
        sc = MinMaxScaler()
        scaled_input = sc.fit_transform(input_data)

        # Convert the scaled data into LSTM input format (sequence of 50 timesteps)
        lstm_input = np.zeros((1, 50, len(features_col_name)))
        lstm_input[0, -1, :] = scaled_input  # Only using the last time step for now

        # Make prediction
        m_pred = model.predict(lstm_input)
        failure_prob = list(m_pred[-1] * 100)[0]
        st.write(f"Predicted failure probability: {failure_prob:.2f}%")
    else:
        st.error("Model not loaded.")

# Model evaluation is not needed for this version, so you can remove it or leave it for later if you want to evaluate with test data.
