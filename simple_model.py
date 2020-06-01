#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:31:57 2020

@author: prasanthchakka
"""

import pandas as pd

df = pd.read_csv('/Users/prasanthchakka/Documents/MBD_term 3/electives/RiskNFraud/Individual assignment/Python-JupyterNotebook/dev.csv')

dfo = pd.read_csv("/Users/prasanthchakka/Documents/MBD_term 3/electives/RiskNFraud/Individual assignment/Python-JupyterNotebook/oot0.csv")

print ("STEP 1: DOING MY TRANSFORMATIONS...")
df = df.fillna(0)
dfo = dfo.fillna(0)

print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")
in_model = ['ib_var_1','icn_var_22','ico_var_25','if_var_65','if_var_77']

output_var = 'ob_target'

print ("STEP 3: DEVELOPING THE MODEL...")
X = df[in_model]
y = df[output_var]
Xo = dfo[in_model]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs')
fitted_model = clf.fit(X, y)



