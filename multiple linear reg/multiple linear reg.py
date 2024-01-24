import numpy as np
import pandas as pd

def betas(x,y):
    betas=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return betas
def fit_model (x,betas):
    y_hat=x.dot(betas)
    return y_hat
def predict(x,betas):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    prediction=x.dot(betas)
    return prediction
def sum_square_error(y,y_hat):
    error =y-y_hat
    sse=(error).dot(error.T)
    return sse
def transform_x (x  , degree):
    temp=x.copy()
    for i in range (2 , degree+1):
        x=np.c_[x, temp**i]
    return x
    
    
    



data =pd.read_csv(r"C:\Users\User\Desktop\linear regression\multiple linear reg\real_data.csv")
# print (data.head())
x=data[['size' , 'year']]
n=len(x)
x_bias=np.ones((n,1))
x=np.c_[x_bias , x]
y=data['price']
betas=betas(x,y)
print (betas)

# ==================================================
sse=sum_square_error(y, fit_model(x,betas))
sst=sum_square_error(y,y.mean())
r_square=(sst-sse)/sst
print (r_square)















        
    