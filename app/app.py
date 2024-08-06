import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(page_title="Air Quality Prediction 🌫️", page_icon="🌟")

# Load the best model
best_model_path = "E:/Project and Advices/Project Air Quality Index project/models/best_model.pkl"
best_model = joblib.load(best_model_path)

# Sidebar for user input
st.sidebar.header("User Input Parameters 🗓️⏰")
selected_date = st.sidebar.date_input("Select Date", datetime.date.today())
selected_time = st.sidebar.time_input("Select Time", datetime.datetime.now().time())

# Extract features from user input
selected_datetime = datetime.datetime.combine(selected_date, selected_time)
month = selected_datetime.month
day = selected_datetime.day
hour = selected_datetime.hour
weekday = selected_datetime.weekday()

# Create input DataFrame for prediction
input_data = pd.DataFrame([[month, day, hour, weekday]], columns=['month', 'day', 'hour', 'weekday'])
prediction = best_model.predict(input_data)

# Display prediction with color and hover effect
st.write(f"<h1 style='color: #007bff;'> Lahore Air Quality Index Predicted </h1> <br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="font-size: 24px; border: 2px solid #007bff; border-radius: 8px; padding: 10px; display: inline-block;">
        <span style="font-weight: bold; color: #007bff;">Predicted PM2.5 Value:</span>
        <span class="hover-effect" style="color: #ff4500; font-weight: bold;"> {prediction[0]} µg/m³</span>
    </div>
    <br>
    <br>
    <style>
        .hover-effect:hover {{
            background-color: #e7f0ff;
        }}
    </style>
    """, unsafe_allow_html=True
)

# Map visualization
def plot_map():
    # Create a folium map centered on Lahore
    lahore_map = folium.Map(location=[31.5497, 74.3436], zoom_start=12)

    # Add a marker for Lahore
    folium.Marker(
        location=[31.5497, 74.3436],
        popup="Lahore City",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(lahore_map)

    # Display the map in Streamlit
    st.write("## Location of Lahore City 🗺️")
    st_folium(lahore_map, width=700, height=500)

plot_map()

# Inject HTML and CSS for styling
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>🌫️ Air Quality Prediction</title>
    <style>
        body {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .stApp {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #007bff;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stSidebar {
            background-color: #007bff;
            color: white;
        }
    </style>
</head>
<body>
    <div class="stApp">
        <!-- Streamlit app content will be injected here -->
    </div>
</body>
</html>
"""
components.html(html_code, height=0, width=0)
