"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics

This file is a simple demonstration of how to use the data interrogation/model 
construction/model evaluation files on your own and set hyperparameters. The data
file is the 3D example used in the paper.
"""

import numpy as np
import pandas as pd
from dataset_property_new import nonlinearity_assess, collinearity_assess, residual_analysis
import cv_final as cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load data
Data = pd.read_excel('Printer.xlsx', header = None).values
X = Data[:, :9] # TODO: One of the columns is being ignored and I do not know why
y = Data[:, 10].reshape(-1, 1)
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

xticks = ['LH','WT','ID','IP','NT','BT','PS','M','FS']
yticks = ['TS']

nonlinearity_assess(X, y, True, [0,0,0,1,0,0,0,1,0], xticks = xticks, yticks = yticks)
collinearity_assess(X, y, True, xticks, yticks)

"""               Model construction and evaluation
Based on the data interrogation result, we select the static nonlinear model.
Here, we build several models to illustrate how to use the model construction.
This file is just an example. It is not the optimal way to build models. """

# ALVEN: The ALVEN model file takes unscaled data, but the final output is scaled
ALVEN_hyper, ALVEN_model, ALVEN_params, mse_train_ALVEN, mse_test_ALVEN, yhat_train, yhat_test, MSE_validation, final_list = cv.CV_mse('ALVEN', X, y, X_test, y_test, cv_type = 'Re_KFold', alpha_num = 30, degree = [1,2], label_name = True)

scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y)
y_test_scaled = scaler_y.transform(y_test)
residual_analysis(X_test, y_test_scaled, yhat_test, alpha = 0.01, round_number = 'ALVEN')

# SVR: Models in SPA, other than ALVEN / DALVEN, require scaling
scaler_x = StandardScaler(with_mean=True, with_std=True)
scaler_x.fit(X)
X = scaler_x.transform(X)
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y)
y = scaler_y.transform(y)
y_test_scaled = scaler_y.transform(y_test)

Nr = 10
SVR_hyper, SVR_model, mse_train_SVR, mse_test_SVR, yhat_train, yhat_test, MSE_validate = cv.CV_mse('SVR', X, y, X_test, y_test_scaled, cv_type = 'Re_KFold', C = [0.001, 1 , 100])

yhat_test = scaler_y.inverse_transform(yhat_test)
residual_analysis(X_test, y_test, yhat_test, alpha = 0.01, round_number = 'SVR')

