import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model


# Load the model
model = load_model('model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the label encoder
le = pickle.load(open('label_encoder_gender.pkl', 'rb'))

# Load the one hot encoder
ohe = pickle.load(open('one_hot_encoder_geography.pkl', 'rb'))

# streamlit app

st.title("Customer Churn Prediction")

# Input fields for customer details
customer_id = st.number_input("Customer ID")
gender = st.selectbox("Gender", le.classes_)
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_of_products = st.number_input("Number of Products")
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary")
geography = st.selectbox("Geography", ohe.categories_[0])
credit_score = st.number_input("Credit Score")


# Create a DataFrame with the input values
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})
# one hot encode geography col
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# combine the dataframes
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the data

input_data_scaled = scaler.transform(input_data)

# predict
prediction = model.predict(input_data_scaled)

if prediction[0][0] > 0.5:
    st.write(prediction[0][0])
    st.write("Customer is likely to churn")
else:
    st.write(prediction[0][0])
    st.write("Customer is not likely to churn")