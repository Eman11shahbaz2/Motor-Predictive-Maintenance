import streamlit as st
import scipy.io
import numpy as np
import joblib

# Load the saved AI model
model = joblib.load('motor_rf_model.pkl')

st.title("⚙️ Motor Predictive Maintenance")
st.write("Upload a CWRU vibration dataset (.mat) to detect bearing faults.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a .mat file", type=["mat"])

if uploaded_file is not None:
    # Load the uploaded .mat file
    mat_data = scipy.io.loadmat(uploaded_file)
    
    # Find the Drive End (DE) vibration array automatically
    de_key = [key for key in mat_data.keys() if 'DE_time' in key]
    
    if len(de_key) > 0:
        vibration_data = mat_data[de_key[0]].flatten()
        
        # Take a 2048-point chunk to test
        chunk = vibration_data[:2048]
        
        # Extract features exactly as we did in the notebook
        features = np.array([[
            np.max(chunk),
            np.min(chunk),
            np.mean(chunk),
            np.std(chunk),
            np.var(chunk)
        ]])
        
        # Ask the model to predict
        prediction = model.predict(features)
        
        st.subheader("Diagnostic Result:")
        if prediction[0] == 0:
            st.success("✅ The motor is NORMAL and healthy.")
        else:
            st.error("🚨 FAULT DETECTED: Broken bearing suspected.")
            
        st.line_chart(chunk) # Plot the waveform for the engineer to see
    else:
        st.error("Could not find Drive End (DE) data in this file.")