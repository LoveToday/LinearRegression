# 多变量线性回归
# 案例：假设你现在打算卖房子，想知道房子能卖多少钱

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


data = pd.read_csv('source/ex1data2.txt', names=['size','bedrooms','price'])
data.head()

# 归一化 方法一 z = (x - u)/delta   方法二 z = (x - min(x))/(max(x) - min(x)) 量化后的区间将分布在区间


def normalize_feature(data):
    return (data - data.mean())/data.std()

data = normalize_feature(data)

print(data.head())

# data.plot.scatter('size', 'price', label='size')
# plt.show()

# 添加全为一的一列
data.insert(0, 'ones', 1)
print(data.head())

# 构造数据集

X = data.iloc[:, 0:-1]
X = X.values

y = data.iloc[:,-1]
y = y.values

y = y.reshape(47,1)

# 损失函数
def costFunction(X,y,theta):
    inner = np.power(X @ theta - y, 2)
    return np.sum(inner) / (2 * len(X))

theta = np.zeros((3,1))
cost_init = costFunction(X,y,theta)

# 梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha/len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        if i % 100 == 0:
            print(cost)

    return theta, costs


# theta, costs = gradientDescent(X, y, theta, alpha=0.05, iters=1000)
iters = 2000
fig,ax = plt.subplots()
# 不同alpha下的效果比较
alphas = [0.0003,0.003,0.03,0.0001,0.001, 0.05]
for aa in alphas:
    _, cs = gradientDescent(X, y, theta, aa, iters)
    ax.plot(np.arange(iters), cs, label=aa)
    ax.legend()

ax.set(xlabel='iters', ylabel='cost', title='cost vs iters')
plt.show()


