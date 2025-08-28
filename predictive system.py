# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

#Loading The Save Model
loaded_model = pickle.load(open('E:/Deploy_Model/trained_model.sav','rb'))

# Input data (example values)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Making prediction
prediction = loaded_model.predict(input_data_reshaped)
print("Raw Prediction:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
