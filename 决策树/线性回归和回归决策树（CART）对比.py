"""
线性回归和回归决策树（CART）对比

CART分类回归决策树，既可以做分类也可以做回归，一般做分类
做分类采用基尼值，做回归采用平方损失,即最小二乘
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor      #分类决策树
from sklearn.linear_model import LinearRegression


#1.准备数据
x_train = np.array(list(range(1,11))).reshape(-1,1)
y_train = np.array([5.65,4.43,6.43,7.5,8.3,9.1,10.5,11.3,12.8,13.6])

#2.模型训练
estimator1 = LinearRegression()                     #线性回归
estimator2 = DecisionTreeRegressor(max_depth=1)     #回归决策树，最大树深度为1
estimator3 = DecisionTreeRegressor(max_depth=3)     #回归决策树，最大树深度为3

estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)

#3.模型预测
x_test = np.arange(0,10,0.1).reshape(-1,1)

y_pre1 = estimator1.predict(x_test)
y_pre2 = estimator2.predict(x_test)
y_pre3 = estimator3.predict(x_test)

#4.绘图
plt.scatter(x_train, y_train, c='gray')
plt.plot(x_test, y_pre1, c='red', label='LinearRegression')
plt.plot(x_test, y_pre2, c='blue', label='DecisionTreeRegressor(max_depth=1)')
plt.plot(x_test, y_pre3, c='green', label='DecisionTreeRegressor(max_depth=3)')
plt.legend()
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.show()