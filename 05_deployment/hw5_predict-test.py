#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict_sub'

client = {"job": "student", "duration": 280, "poutcome": "failure"}


response = requests.post(url, json=client).json()
print(response)

print('The probability that this client will get a subscription is',
      round(response['subscription_probability'], 3))
