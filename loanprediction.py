# import pandas as pd
# loan_pd = pd.read_csv('loan_approval_dataset.csv')
# loan_pd

# loan_pd.isnull().sum()

# loan_pd_shuffled = loan_pd.sample(n = len(loan_pd), axis = 0 , random_state = 1)
# loan_pd_shuffled

# loan_pd_shuffled.dtypes

# a = loan_pd_shuffled.select_dtypes(include = ['object']).copy()
# a

# a[' education'].value_counts()
# a.isnull().sum()

# a[' education'] = a[' education'].map({" Graduate" : 1 , " Not Graduate" : 0})
# a
# a[' loan_status'] = a[' loan_status'].map({" Rejected" : 0 , " Approved" : 1})
# a
# a[' self_employed'] = a[' self_employed'].map({" Yes" : 1 , " No" : 0})
# a
# a.dropna()
# a

# a.dtypes

# a

# loan_pd_final = pd.concat([loan_pd_shuffled.drop([' education' , ' self_employed' , ' loan_status'], axis = 1) , a], axis = 1)

# loan_pd_final


# X = loan_pd_final.iloc[: , :-1]
# y = loan_pd_final.iloc[:,-1]

# from sklearn.model_selection import train_test_split
# X_train ,X_test ,y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 20)

# print(X_train.dtypes)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X_train.shape, X_test.shape

# pd.DataFrame(X_train).head()

# # from sklearn.metrics import mean_squared_error as mse
# from sklearn.linear_model import LinearRegression

# lm = LinearRegression().fit(X_train, y_train)
# #mse(lm.predict(X_train), y_train, squared=False), mse(lm.predict(X_val), y_val, squared=False)
#   # R-squared on training data
# train_score = lm.score(X_train, y_train)

# train_score,

# from sklearn.neighbors import KNeighborsRegressor

# knn = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
# train_score = knn.score(X_train, y_train)
# train_score

# from sklearn.ensemble import RandomForestRegressor

# rfr = RandomForestRegressor(max_depth=10).fit(X_train, y_train)
# train_score = rfr.score(X_train, y_train)
# train_score

# rfr_prediction = rfr.predict(X_test)

# print("Predicted value of y " , rfr_prediction)
# print("Actual value of y" , y_test)

# from sklearn.linear_model import LogisticRegression
# lgrm = LogisticRegression()
# lgrm.fit(X_train , y_train )
# train_score = lgrm.score(X_train , y_train)
# train_score,

import streamlit as st 
st.write("hello world")






