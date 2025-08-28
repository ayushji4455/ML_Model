# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 23:56:30 2025

@author: Ayush
"""

import numpy as np
import pickle
import streamlit

#Loading The Save Model
loaded_model = pickle.load(open('E:/Deploy_Model/trained_model.sav','rb'))
