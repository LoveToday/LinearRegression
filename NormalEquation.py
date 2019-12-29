# 正规方程

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('source/ex1data1.txt', names=['population','profit'])
data.insert(0,'ones',1)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

X = X.values 
y = y.values
y = y.reshape(97,1)


def normalEquation(X, y):
    # 求逆
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta
theta = normalEquation(X,y)
print(theta)