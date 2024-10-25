import sklearn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import time  # Import time module

st.title('California Housing Predictor')
name = st.text_input("Enter your Name")

if st.button('Click Me'):
    st.write('Button Clicked')



californi_housing = fetch_california_housing()

from PIL import Image
image = Image.open("C:\Users\jhapr\Downloads\californiahousingsampleimage1.jpg" , )
st.image(image , caption="house", use_column_width=True)


#Created a StreamLit UI

user_input_neighbors = st.sidebar.slider('Enter the number of neighbors', 1, 10, 3)
user_input_population = st.sidebar.slider('Enter the Population', 0, int(df["Population"].max()), 1000)
user_input_bedrooms = st.sidebar.slider('Enter the Average Bedrooms', 0, int(df["AveBedrms"].max()), 3)
user_input_occupancy = st.sidebar.slider('Enter the Average Occupancy', 0, int(df["AveOccup"].max()), 3)
user_input_rooms = st.sidebar.slider('Enter the Average Rooms', 0, int(df["AveRooms"].max()), 5)
user_input_age = st.sidebar.slider('Enter the House Age', 0, int(df["HouseAge"].max()), 20)
user_input_income = st.sidebar.slider('Enter the Median Income', 0, int(df["MedInc"].max()), 4)