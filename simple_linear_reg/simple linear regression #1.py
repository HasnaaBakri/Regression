# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:42:18 2022

@author: HASSNA
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def compute_b1(x,x_bar ,y, y_bar):
    first_sum=0
    second_sum=0
    for i in range(len(x)): 
        first_sum+=(x[i] - x_bar)*(y[i] - y_bar)
        second_sum+=(x[i] - x_bar) ** 2
    return first_sum/second_sum

def compute_b0(x_bar , y_bar , b1):
    return y_bar-b1*x_bar

def compute_square_error(x,y,b0,b1):
    ssr=0
    for i in range(len(x)):
        ssr+=(y[i]-(b0+b1*x[i]))**2
    return ssr

def y_hat(x,b0,b1):
    n=len(x)
    y_hat=np.zeros(n)
    for i in range (n):
        y_hat[i]=b0+b1*x[i]
    return y_hat

def predict(x, b0,b1):
    return b0+b1*x

data = pd.read_csv(r"C:\Users\User\Desktop\linear regression\simple_linear_reg\real_data.csv")
x = data["size"]
y = data["price"]

x_bar=np.mean(x)
y_bar=np.mean(y)
plt.scatter(x,y)
plt.show()

b1=compute_b1(x, x_bar, y, y_bar)
b0=compute_b0(x_bar, y_bar, b1)
y_hat=y_hat(x,b0,b1)
plt.scatter(x,y)
plt.plot(x,y_hat)
plt.show()
print (compute_square_error(x,y,b0,b1))
print (predict(643.09, b0, b1))
print ("===============================================")
sse=compute_square_error(x,y,b0,b1)
sst=compute_square_error(x,y,np.mean(y),0)
ssr=sst-sse
r_square=(ssr/sst)*100
# print (sst)
# print (sse)
print (r_square)
print (b0,b1)



























        
    