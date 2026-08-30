"""
案例：
    线性回归对象，波士顿房价预测（正规方程法）

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
"""
from sklearn.preprocessing import StandardScaler        #数据预处理
from sklearn.linear_model import LinearRegression       #正规方程的线性回归
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error          #均方误差
import pandas as pd
import numpy as np


#1.加载波士顿房价数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
#print(f"特征：{data.shape}, 标签：{target.shape}")
#print(f"特征数据：{data[:3]}，标签数据：{target[:3]}")

#2.数据预处理
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=23)

#3.特征工程
#3.1创建标准化对象
transfer = StandardScaler()
#3.2对训练集进行训练并转化
x_train = transfer.fit_transform(x_train)
#3.3对测试集进行转化
x_test = transfer.transform(x_test)

#4.模型训练
estimator = LinearRegression(fit_intercept=True)        #fit_intercept: 是否创建偏置项，默认为True
estimator.fit(x_train,y_train)
print(f'权重为：{estimator.coef_}')
print(f'偏置为：{estimator.intercept_}')

#5.模型预测
y_pre = estimator.predict(x_test)
print(f'预测结果：{y_pre}')

#6.模型评估
#参1：测试集的标签，参2：预测值
print(f'均方误差：{mean_squared_error(y_test,y_pre)}')           #MSE：均方误差，公式：每个样本的误差的平方和除以样本数
print(f'均方根误差：{root_mean_squared_error(y_test,y_pre)}')    #RMSE：均方根误差，公式：每个样本的误差的平方和除以样本数的平方根
print(f'平均绝对误差：{mean_absolute_error(y_test,y_pre)}')      #MAE：平均绝对误差，公式：每个样本的误差的绝对值和除以样本数