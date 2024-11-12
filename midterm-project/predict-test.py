#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

patient_id = 'xyz-543'
patient = {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31,
    "BMI_Category_Normal": 'false',
    "BMI_Category_Overweight": 'true',
    "BMI_Category_Obese": 'false'}

patient2 = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
    "BMI_Category_Normal": 'false',
    "BMI_Category_Overweight": 'false',
    "BMI_Category_Obese": 'true'}

response = requests.post(url, json=patient2).json()
print(response)

if response['diabetes'] == True:
    print('Sending an appointment email to %s' % patient_id)
else:
    print('NOT sending an appointment email to %s' % patient_id)
