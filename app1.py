import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# Load the model
model = joblib.load('road_classifier_model.pkl')

st.set_page_config(page_title="AI Map Matcher", layout="wide")
st.title("🛣️ Vehicular Movement Classifier")
st.subheader("Distinguishing Highways vs. Service Roads using AI-ML")

# --- Sidebar Inputs ---
st.sidebar.header("Real-time GPS Input")
speed = st.sidebar.slider("Vehicle Speed (km/h)", 0, 150, 80)
dist_hwy = st.sidebar.slider("Distance to Highway (m)", 0, 100, 5)
dist_srv = st.sidebar.slider("Distance to Service Road (m)", 0, 100, 30)
accel = st.sidebar.number_input("Acceleration (m/s²)", -5.0, 5.0, 0.5)

# --- Prediction Logic ---
features = pd.DataFrame([[speed, accel, 0, 0, dist_hwy, dist_srv]], 
                        columns=['speed', 'acceleration', 'heading_change', 'stop_frequency', 'dist_to_hwy', 'dist_to_srv'])

prediction = model.predict(features)[0]
prob = model.predict_proba(features).max() * 100

# --- Display Results ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted Road Type", value=prediction.replace("_", " ").title())
    st.write(f"**Confidence Score:** {prob:.2f}%")
    
    if prediction == "highway":
        st.success("Vehicle is moving at high speed on the main corridor.")
    else:
        st.warning("Vehicle is moving slowly/interrupted on a service road.")

with col2:
    # Small Map Visualization
    m = folium.Map(location=[28.50, 77.07], zoom_start=15)
    # Add a marker representing the vehicle
    color = "blue" if prediction == "highway" else "red"
    folium.Marker([28.50, 77.07], tooltip="Vehicle Location", icon=folium.Icon(color=color)).add_to(m)
    st_folium(m, height=300, width=500)