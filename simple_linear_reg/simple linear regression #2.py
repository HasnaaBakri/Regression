import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_model(x, b):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    y_hat = x.dot(b)
    return y_hat


def find_betas(x, y):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    betas = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return betas


def predict(x, betas):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    prediction = x.dot(betas)
    return prediction


data = pd.read_csv(
    r"C:\Users\User\Desktop\linear regression\simple_linear_reg\real_data.csv")

x = data[['size', 'year']]
y = data['price']

betas = find_betas(x, y)
print(betas)
print(predict([[650, 2017]], betas))