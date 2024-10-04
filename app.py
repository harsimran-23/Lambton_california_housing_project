import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from  sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# load the dataset
cal= fetch_california_housing()
df= pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['price']= cal.target
df.head()

# title of the app 
st.title("california house price prediction for XYZ comapny")   

#data overview 
st.subheader("Data Overview")
st.write(df.head(10))

# split the data into train and test 
x= df.drop(['price'], axis=1) # input variables 
y=df['price'] #target variables
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=42)

#standardize the data 
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)

#Model Selection
st.subheader('Select the model')
model = st.selectbox("Choose a model",["Linear Regression", "Ridge", "Lasso", "Elastic net"])

#initialize the model 

models = {
    "Linear Regression":LinearRegression(),
    "Ridge": Ridge(),
    "Lasso":Lasso(),
    "Elastic net": ElasticNet()}


#train the selected model  
selected_model= models[model]


#train the model 
selected_model.fit(x_train_sc, y_train)

#predict the values 
y_pred = selected_model.predict(x_test_sc)

#evaluate the model using metrics


test_mse=mean_squared_error(y_test,y_pred)
test_mae= mean_absolute_error(y_test,y_pred)
test_rmse= np.sqrt(test_mse)
test_r2= r2_score(y_test,y_pred)


#Display the metrics

st.write("test_mse",test_mse)
st.write("test_mae",test_mae)
st.write("test_rmse",test_rmse)
st.write("test_r2",test_r2)

#prompt the user to enter the input vals 
st.write("Ënter the values to predit the house price")

user_input={}

for feature in x.columns:
    user_input[feature]= st.number_input(feature)

#create a dataframe
user_input_df= pd.DataFrame([user_input])

#scale the use input
user_input_sc = scaler.transform(user_input_df)

#predit the house price 
predicted_price = selected_model.predict(user_input_sc)

#display the predicted house price

st.write(f"predicted house price is {predicted_price[0]*100000}")

