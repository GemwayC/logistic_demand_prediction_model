import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
from geopy.distance import geodesic
import datetime

# Load your trained model (after saving it using pickle)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the page with a wide layout and a background color
st.set_page_config(page_title="Delivery Demand Forecasting", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True
)

# Page title and description with custom formatting
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Daily Delivery Demand Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the number of deliveries based on input features like weather, traffic, agent information, and more.</p>", unsafe_allow_html=True)

# Section header for input features
st.markdown("<h2 style='color: #4682B4;'>Input Details</h2>", unsafe_allow_html=True)

# Create input fields in two columns for better layout
col1, col2 = st.columns(2)

with col1:
    agent_age = st.number_input('Agent Age', min_value=18, max_value=65, value=None, format="%d")
    agent_rating = st.slider('Agent Rating', min_value=1.0, max_value=5.0, value=None, step=0.1)
    weather = st.selectbox('Weather Condition', ['Select an option', 'fog', 'Cloudy', 'sandstorms', 'Stormy', 'Sunny', 'Windy'])
    traffic = st.selectbox('Traffic Condition', ['Select an option', 'Low', 'Medium', 'High', 'Jam'])
    vehicle = st.selectbox('Vehicle Type', ['Select an option', 'Motorcycle', 'Scooter', 'Van'])

with col2:
    area = st.selectbox('Area Type', ['Select an option', 'Urban', 'Semi-Urban', 'Metropolitan', 'Other'])
    category = st.selectbox('Category', ['Select an option', 'Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys',
        'Snacks','Shoes', 'Apparel', 'Jewelry', 'Outdoors', 'Grocery', 'Books', 'Kitchen', 
        'Home', 'Pet_Supplies', 'Skincare'])
    store_latitude = st.number_input('Store Latitude', value=None)
    store_longitude = st.number_input('Store Longitude', value=None)
    drop_latitude = st.number_input('Drop-off Latitude', value=None)
    drop_longitude = st.number_input('Drop-off Longitude', value=None)

# Input for Order Date and Time
order_date = st.date_input('Order Date', value=datetime.date.today())
order_time = st.time_input('Order Time', value=datetime.time(12, 0))  # Default to noon
pickup_time = st.time_input('Pickup Time', value=datetime.time(12, 0))  # Default to noon

# Only calculate distance and make predictions if all fields are filled
if st.button('Predict Delivery Demand'):
    # Check if any required fields are empty
    if None in [agent_age, agent_rating, weather, traffic, vehicle, area, category, store_latitude, store_longitude, drop_latitude, drop_longitude]:
        st.error("Please fill in all fields before making a prediction.")
    else:
        # Calculate the distance between store and drop-off
        distance = geodesic((store_latitude, store_longitude), (drop_latitude, drop_longitude)).km

        # Extract time components
        order_hour = order_time.hour
        pickup_hour = pickup_time.hour
        
        # Date input and feature extraction
        day_of_week = order_date.weekday()
        month = order_date.month

        # Encode categorical inputs as numbers (same as used in the training)
        weather_mapping = {'fog': 0, 'Cloudy': 1, 'sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5}
        traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
        vehicle_mapping = {'Motorcycle': 0, 'Scooter': 1, 'Van': 2}
        area_mapping = {'Urban': 0, 'Semi-Urban': 1, 'Metropolitan': 2, 'Other': 3}
        category_mapping = {'Clothing':0, 'Electronics':1, 'Sports':2, 'Cosmetics':3, 'Toys':4,
        'Snacks':5,'Shoes':6, 'Apparel':7, 'Jewelry':8, 'Outdoors':9, 'Grocery':10, 'Books':11, 'Kitchen':12, 
        'Home':13, 'Pet_Supplies':14, 'Skincare':15}

        # Prepare the data for prediction
        input_data = np.array([[agent_age, agent_rating, order_hour, pickup_hour, day_of_week, month, distance,
                                weather_mapping[weather], traffic_mapping[traffic],
                                vehicle_mapping[vehicle], area_mapping[area], category_mapping[category]]])

        # Make the prediction
        prediction = model.predict(input_data)
        st.success(f'Predicted Daily Deliveries: {int(prediction[0])}')
        st.balloons()

# Footer section with contact info or credits
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey;'>
    Created by Florence Idowu for Gemway Consult | Powered by Streamlit
    </p>
    """, unsafe_allow_html=True
)