from tensorflow.keras.models import load_model
model = load_model("model2.h5")
import numpy as np
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
with open("label_encoder.pkl","rb") as file:
    label = pickle.load(file)


with open("oneHot.pkl","rb") as file:
    oneHot = pickle.load(file)



with open("scalar.pkl" , "rb") as file:
    scalar = pickle.load(file)      


model = load_model("model2.h5")
st.title('Customer Churn Prediction')

geography = st.selectbox("Geography",oneHot.categories_[0])
gender = st.selectbox("Gender",label.classes_)
age = st.slider("Age",18,92)
balance= st.number_input("Balance")
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore':[credit_score],
    'Gender': [gender],
    'Age': [age],
    'Geography': [geography],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}
input_df = pd.DataFrame(input_data)

input_df["Gender"]= label.transform(input_df['Gender'])

geo_one = oneHot.transform(input_df[["Geography"]])
geo_dataFrame = pd.DataFrame(geo_one,columns=oneHot.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df.drop("Geography",axis=1),geo_dataFrame],axis=1)

input_scaled = scalar.transform(input_df)
prediction = model.predict(input_scaled)
churn = prediction[0][0]

if churn > 0.5:
    st.write('Customer is likely to churn.')
else:
    st.write("Customer is not likely to churn.") 

st.write(f'Churn probability: {churn}')
   




