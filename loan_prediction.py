import sklearn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import time  # Import time module

import pandas as pd
loan_pd = pd.read_csv('loan_approval_dataset.csv')
# loan_pd
loan_pd_shuffled = loan_pd.sample(n = len(loan_pd), axis = 0 , random_state = 1)
# loan_pd_shuffled
a = loan_pd_shuffled.select_dtypes(include = ['object']).copy()
a[' education'] = a[' education'].map({" Graduate" : 1 , " Not Graduate" : 0})
a[' loan_status'] = a[' loan_status'].map({" Rejected" : 0 , " Approved" : 1})
a[' self_employed'] = a[' self_employed'].map({" Yes" : 1 , " No" : 0})
a.dropna()
loan_pd_final = pd.concat([loan_pd_shuffled.drop([' education' , ' self_employed' , ' loan_status'], axis = 1) , a], axis = 1)


X = loan_pd_final.iloc[: , :-1]
X = X.drop(columns=['loan_id'])
y = loan_pd_final.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 20)

# print(X_train.dtypes)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train.shape, X_test.shape

# pd.DataFrame(X_train).head()

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=10).fit(X_train, y_train)
train_score = rfr.score(X_train, y_train)
train_score

st.markdown('''**Loan Prediction System** ''')

import PIL 


#Created a StreamLit UI

# user_input_neighbours = st.sidebar.slider('Enter the number of neighbors', 1, 10, 3)
user_input_no_of_dependents = st.sidebar.slider('Enter the no_of_dependents', 0, int(loan_pd_final[" no_of_dependents"].max()))
user_input_income_annum = st.sidebar.slider('Enter the income_annum', 0, int(loan_pd_final[" income_annum"].max()), 3)
user_input_loan_amount = st.sidebar.slider('Enter the loan_amount', 0, int(loan_pd_final[" loan_amount"].max()), 3)
user_input_loan_term = st.sidebar.slider('Enter the loan_term', 0, int(loan_pd_final[" loan_term"].max()), 5)
user_input_cibil_score = st.sidebar.slider('Enter the cibil_score', 0, int(loan_pd_final[" cibil_score"].max()), 20)
user_input_residential_assets_value = st.sidebar.slider('Enter the residential_assets_value', 0, int(loan_pd_final[" residential_assets_value"].max()), 4)
user_input_commercial_assets_value = st.sidebar.slider('Enter the commercial_assets_value', 0, int(loan_pd_final[" commercial_assets_value"].max()))
user_input_luxury_assets_value = st.sidebar.slider('Enter the luxury_assets_value', 0, int(loan_pd_final[" luxury_assets_value"].max()))
user_input_bank_asset_value = st.sidebar.slider('Enter the bank_asset_value', 0, int(loan_pd_final[" bank_asset_value"].max()))

# st.write(user_input_no_of_dependents)

options = ["Graduate", "Not Graduate"]
selectbox_selection_edu = st.selectbox("Education", options)
# st.write(f"Color selected is {selectbox_selection}")

options = ["Yes", "No"]
selectbox_selection_selfemp = st.selectbox("Self_Employed ", options)
# st.write(f"Color selected is {selectbox_selection}")


# Create a dataframe for user Input 
user_input_df = pd.DataFrame({
    ' no_of_dependents': [user_input_no_of_dependents],
    ' income_annum': [user_input_income_annum],
    ' loan_amount': [user_input_loan_amount],
    ' loan_term': [user_input_loan_term],
    ' cibil_score': [user_input_cibil_score],
    ' residential_assets_value': [user_input_residential_assets_value],
    ' commercial_assets_value': [user_input_commercial_assets_value],
    ' luxury_assets_value': [user_input_luxury_assets_value],
    ' bank_asset_value': [user_input_bank_asset_value],
    ' education': [selectbox_selection_edu],
    ' self_employed': [selectbox_selection_selfemp]
}, columns = X.columns)

# X

# user_input_df[' education'] = user_input_df[' education'].map({"Graduate" : 1 , "Not Graduate" : 0})
# user_input_df[' self_employed'] = user_input_df[' self_employed'].map({"Yes" : 1 , "No" : 0})

b = user_input_df.select_dtypes(include = ['object']).copy()
b[' education'] = b[' education'].map({"Graduate" : 1 , "Not Graduate" : 0})
b[' self_employed'] = b[' self_employed'].map({"Yes" : 1 , "No" : 0})
b.dropna()
user_input_df = pd.concat([user_input_df.drop([' education' , ' self_employed'], axis = 1) , b], axis = 1)

user_input_df

#Scaling the user input dataframe 
user_input_scaled = scaler.transform(user_input_df)

predicted_loan_status = rfr.predict(user_input_scaled)

predicted_loan_status

if int(predicted_loan_status) == 1 :
    st.write("The predicted loan status is  Approved")
if int(predicted_loan_status) == 0 :
    st.write("The predicted loan status is Not Approved")


