# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:51:34 2023

@author: Indoskill
"""

import pickle as pkl
import numpy as np
import streamlit as st

filename = "C:/Users/Indoskill/0. Saved_Models/Diabetes Prediction/diabetes_model.sav"
load_model = pkl.load(open(filename, 'rb'))

st.title('Diabetes Disease Prediction')
#input_data = st.number_input("Enter the Temperarture Value: ", max_value=100, value=0)
#result = st.write("The prediction is")

def predictions(data):
    data_array = np.asarray(data).reshape(1,-1)
    pred = load_model.predict(data_array)
    if pred[0] == 0:
        return "The person is Non Diabetic"
    else:
        return "The person is having Diabetes"

def main():
    #input_data = 12
    #print(predictions(input_data))
    pr = st.number_input("No. of Pregnencies: ", value=0.0)
    gl = st.number_input("Glucose Level: ",  value=0.0)
    bp = st.number_input("Blood Pressure: ", value=0.0)
    sk = st.number_input("Skin Thickness:", value=0.0)
    _in = st.number_input("Insulin:", value=0.0)
    bmi = st.number_input("BMI", value=0.0)
    dbf = st.number_input("Diabetes Pedigree Function:", value=0.0)
    ag = st.number_input("Age:", value=0.0)
    input_data = [pr,gl,bp,sk,_in,bmi,dbf,ag]
    
    if st.button("Predict"):
        result = predictions(input_data)
        st.write(result)
if __name__ == "__main__":
    main()
    
