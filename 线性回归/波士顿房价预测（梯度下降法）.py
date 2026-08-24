"""
案例：
    线性回归对象，波士顿房价预测（梯度下降法）

    线性回归算法属于：有监督学习，即有特征，有标签，且标签是连续的
    线性回归分类：
        1.一元线性回归：一个特征，一个标签
        2.多元线性回归：多个特征，一个标签
    公式：
        一元线性回归：
            y = wx + b
        多元线性回归：
            y = w1x1 + w2x2 + ... + wnxn + b
              = w的转置 * x + b
公式：
    Θ₂ = Θ₁ - α * ∇Θ₁
    α是学习率
"""

from sklearn.preprocessing import StandardScaler        #数据预处理
from sklearn.linear_model import SGDRegressor           #梯度下降的线性回归
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error          #均方误差

import pandas as pd
import numpy as np


#1.加载波士顿房价数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#2.数据预处理
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=23)

#3.特征工程
sds = StandardScaler()
x_train = sds.fit_transform(x_train)
x_test = sds.transform(x_test)

#4.模型训练
#参1：是否计算截距，参2：学习率模式-->常量，即不会改变，参3：学习率
estimator = SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01)
estimator.fit(x_train,y_train)
print(f'权重为：{estimator.coef_}')
print(f'偏置为：{estimator.intercept_}')

#.模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果：{y_pred}')

#模型评估
print(f'均方误差：{mean_squared_error(y_test,y_pred)}')
print(f'均方根误差：{root_mean_squared_error(y_test,y_pred)}')
print(f'平均绝对误差：{mean_absolute_error(y_test,y_pred)}')
