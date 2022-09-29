import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#read data
path='data2.txt'
data=pd.read_csv(path,header=None , names=['Size' , 'Bedrooms' , 'Price'])

#show data
print ('data \n' , data.head(20))
print('data.describe=',data.describe())

#rescaling data
data=((data -data.mean()) / data.std())

# add ones column
data.insert(0,'Ones',1)

#separate X (training data ) from y (target variable)
cols =data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

#convert to matrices and initialize theta
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))


#cost function
def computeCost(X,y,theta):
    z=np.power(((X*theta.T)-y),2)
    return np.sum(z)/(2*len(X))

#GD function
def gradientDescent (X,y ,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters =int(theta.ravel().shape[1])
    cost =np.zeros(iters)
    for i in range(iters):
        error =(X*theta.T)-y
        for j in range (parameters):
            term =np.multiply(error , X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X)) *np.sum(term))
        theta= temp
        cost[i]=computeCost(X,y,theta)
    return theta , cost
#initialize variables for 
alpha =0.1
iters=100

#
g , cost=gradientDescent(X,y,theta,alpha,iters)
thiscost=computeCost(X,y,g)

#get best fit line for size vs. Price

x=np.linspace(data.Size.min(), data.Size.max(),100)
f=g[0,0]+(g[0,1]*x)

fig , ax=plt.subplots(figsize=(5,5))
ax.plot(x,f,'r','prediction')
ax.scatter (data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

#get best fit line for bedrooms vs. Price

x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)


f = g[0, 0] + (g[0, 1] * x)


# draw the line  for Bedrooms vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

#draw error graph
fig, ax=plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters) , cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
