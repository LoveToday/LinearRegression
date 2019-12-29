# 单变量线性回归
# 案例: 假设你是一家餐厅的CEO, 正在考虑开一家分店，根据该城市的人口数据预测其利润

# 科学计算库，处理多维数组，进行数据分析
import numpy as np
# 是基于numpy的一种工具，该工具是为了解决数据分析任务而创建的
import pandas as pd 
# Python的2D绘图库 matplotlib.pyplot提供一个类似matlab的绘图框架
import matplotlib.pyplot as plt

data = pd.read_csv('source/ex1data1.txt', names=['population', 'profit'])
data.head()

print(data.head())

# 绘制散点图
# data.plot.scatter('population', 'profit', label='population')
# plt.show()

data.insert(0, 'ones', 1)

print(data.head())

X = data.iloc[:,0:-1]

print(X.head())

y = data.iloc[:,-1]
print(y.head())

X = X.values

print(X.shape)

y = y.values

print(y.shape)

y = y.reshape(97, 1)

print(y.shape)

def costFunction(X,y,theta):
    inner = np.power(X @ theta - y, 2)
    return np.sum(inner) / (2 * len(X))
theta = np.zeros((2,1))
print(theta.shape)

cost_init = costFunction(X, y, theta)

def gradientDescent(X, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha/len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        if i % 100 == 0:
            print(cost)

    return theta, costs

alpha = 0.02
iters = 2000


theta, costs = gradientDescent(X, y, theta, alpha, iters)

# 损失函数
# fig, ax = plt.subplots()
# ax.plot(np.arange(iters), costs)
# ax.set(xlabel='iters', ylabel = 'costs' title='cost vs iters')
# plt.show()

# 最小值，最大值，个数
x = np.linspace(y.min(), y.max(),100)
y_ = theta[0,0] + theta[1, 0] * x

fig,ax = plt.subplots();

ax.scatter(X[:,1],y,label='training data')
ax.plot(x, y_, label='predict')
ax.legend()
ax.set(xlabel='population', ylabel='profit')
plt.show()





