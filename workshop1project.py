import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.write('Loan Predictor')



#Data
lad = pd.read_csv('loan_approval_dataset.csv')