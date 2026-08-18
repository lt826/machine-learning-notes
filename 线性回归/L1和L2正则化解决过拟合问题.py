"""
案例：
    演示：欠拟合，正好拟合，过拟合，L1正则化，L2正则化的效果图

    欠拟合：
        模型在训练集上的表现很差，在测试集上的表现也很差
    过拟合：
        模型在训练集上的表现很好，但在测试集上的表现很差

过拟合，欠拟合解释：
    欠拟合：模型过于简单
    过拟合：模型过于复杂

    解决方案：
        欠拟合：增加特征，增加模型的复杂度
        过拟合：手动筛选减少特征，减少模型的复杂度，L1和L2正则化

    L1和L2正则化
        目的：
            都是基于惩罚系数来修改（特征列的）权重，惩罚系数越大，则修改力度越大，对应的权重就越小
        区别：
            L1正则化：惩罚系数越大，对应的权重就越小，甚至会变成0，从而实现特征选择
            L2正则化：惩罚系数越大，对应的权重就越小，无限趋近于0，但不会变成0，因此不会实现特征选择
"""


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler        #数据预处理
from sklearn.linear_model import SGDRegressor           #梯度下降的线性回归
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error          #均方误差
from sklearn.linear_model import Lasso,Ridge            #L1正则化，L2正则化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#定义函数，模拟欠拟合
def demo_under_fitting():
    #准备数据
    #随机生成x轴100个数据，模拟特征
    #指定随机种子，使每次生成（噪声）的数据是相同的
    np.random.seed(10)
    x = np.random.uniform(-3, 3, size=100)      #参1：最小值，参2：最大值，参3：生成个数
    #基于x轴值，通过线性公式，生成y轴100个数据，模拟标签
    #线性公式：y = kx + b = 0.5 * x **2 + x + 0.4 + 噪声
    y = 0.5 * x **2 + x + 0.4 + np.random.normal(0, 1, size=100)    #参1：均值，参2：标准差，参3：生成个数
    #查看生成的x轴和y轴的数据
    #print(f"x轴数据：{x}")
    #print(f"y轴数据：{y}")

    #数据预处理，把x轴转成一列多行的形式
    X = x.reshape(-1, 1)

    #模型训练
    estimator = LinearRegression()
    estimator.fit(X, y)

    #模型预测
    y_predict = estimator.predict(X)
    print(f'模型预测值：{y_predict}')

    #模型评估
    print(f'均方误差：{mean_squared_error(y, y_predict)}')

    #绘图
    plt.scatter(x, y)               #以散点图绘制真实值
    plt.plot(x, y_predict, color='red')             #以折线图绘制预测值
    plt.show()


#定义函数，模拟正好拟合
def demo_just_fitting():
    # 准备数据
    # 随机生成x轴100个数据，模拟特征
    # 指定随机种子，使每次生成（噪声）的数据是相同的
    np.random.seed(10)
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值，参2：最大值，参3：生成个数
    # 基于x轴值，通过线性公式，生成y轴100个数据，模拟标签
    # 线性公式：y = kx + b = 0.5 * x **2 + x + 0.4 + 噪声
    y = 0.5 * x ** 2 + x + 0.4 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数
    # 查看生成的x轴和y轴的数据
    # print(f"x轴数据：{x}")
    # print(f"y轴数据：{y}")

    # 数据预处理，把x轴转成一列多行的形式
    X1 = x.reshape(-1, 1)
    #因为现在只有1个特征，模型过于简单，会出现欠拟合问题，需要增加特征
    X2 = np.hstack([X1, X1 ** 2])      #该函数作用：将X1和x**2水平拼接成新的特征矩阵X2

    # 模型训练
    estimator = LinearRegression()
    estimator.fit(X2, y)

    # 模型预测
    y_predict = estimator.predict(X2)
    print(f'模型预测值：{y_predict}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_predict)}')

    # 绘图
    plt.scatter(x, y)  # 以散点图绘制真实值
    # np.sort: 对x进行排序,默认升序
    # np.argsort: 对x进行排序，返回索引
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以折线图绘制预测值
    plt.show()


#定义函数，模拟过拟合
def demo_over_fitting():
    # 准备数据
    # 随机生成x轴100个数据，模拟特征
    # 指定随机种子，使每次生成（噪声）的数据是相同的
    np.random.seed(10)
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值，参2：最大值，参3：生成个数
    # 基于x轴值，通过线性公式，生成y轴100个数据，模拟标签
    # 线性公式：y = kx + b = 0.5 * x **2 + x + 0.4 + 噪声
    y = 0.5 * x ** 2 + x + 0.4 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数
    # 查看生成的x轴和y轴的数据
    # print(f"x轴数据：{x}")
    # print(f"y轴数据：{y}")

    # 数据预处理，把x轴转成一列多行的形式
    X1 = x.reshape(-1, 1)
    #因为现在只有1个特征，模型过于简单，为了模拟过拟合，我们需要增加9列特征，增加模型的复杂度
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9])

    # 模型训练
    estimator = LinearRegression()
    estimator.fit(X3, y)

    # 模型预测
    y_predict = estimator.predict(X3)
    print(f'模型预测值：{y_predict}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_predict)}')

    # 绘图
    plt.scatter(x, y)  # 以散点图绘制真实值
    # np.sort: 对x进行排序,默认升序
    # np.argsort: 对x进行排序，返回索引
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以折线图绘制预测值
    plt.show()

#定义函数，模拟L1正则化
def demo_l1_regularization():
    # 准备数据
    # 随机生成x轴100个数据，模拟特征
    # 指定随机种子，使每次生成（噪声）的数据是相同的
    np.random.seed(10)
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值，参2：最大值，参3：生成个数
    # 基于x轴值，通过线性公式，生成y轴100个数据，模拟标签
    # 线性公式：y = kx + b = 0.5 * x **2 + x + 0.4 + 噪声
    y = 0.5 * x ** 2 + x + 0.4 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数
    # 查看生成的x轴和y轴的数据
    # print(f"x轴数据：{x}")
    # print(f"y轴数据：{y}")

    # 数据预处理，把x轴转成一列多行的形式
    X1 = x.reshape(-1, 1)
    #因为现在只有1个特征，模型过于简单，为了模拟过拟合，我们需要增加9列特征，增加模型的复杂度
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9])

    # 模型训练
    #estimator = LinearRegression()
    #改为创建L1正则化对象
    estimator = Lasso(alpha= 0.1)    #alpha为惩罚系数，值越大，惩罚力度越大，对应的权重越小
    estimator.fit(X3, y)

    # 模型预测
    y_predict = estimator.predict(X3)
    print(f'模型预测值：{y_predict}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_predict)}')

    # 绘图
    plt.scatter(x, y)  # 以散点图绘制真实值
    # np.sort: 对x进行排序,默认升序
    # np.argsort: 对x进行排序，返回索引
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以折线图绘制预测值
    plt.show()

#定义函数，模拟L2正则化
def demo_l2_regularization():
    # 准备数据
    # 随机生成x轴100个数据，模拟特征
    # 指定随机种子，使每次生成（噪声）的数据是相同的
    np.random.seed(10)
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值，参2：最大值，参3：生成个数
    # 基于x轴值，通过线性公式，生成y轴100个数据，模拟标签
    # 线性公式：y = kx + b = 0.5 * x **2 + x + 0.4 + 噪声
    y = 0.5 * x ** 2 + x + 0.4 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数
    # 查看生成的x轴和y轴的数据
    # print(f"x轴数据：{x}")
    # print(f"y轴数据：{y}")

    # 数据预处理，把x轴转成一列多行的形式
    X1 = x.reshape(-1, 1)
    #因为现在只有1个特征，模型过于简单，为了模拟过拟合，我们需要增加9列特征，增加模型的复杂度
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9])

    # 模型训练
    #estimator = LinearRegression()
    #改为创建L2正则化对象
    estimator = Ridge(alpha= 10)    #alpha为惩罚系数，值越大，惩罚力度越大，对应的权重越小
    estimator.fit(X3, y)

    # 模型预测
    y_predict = estimator.predict(X3)
    print(f'模型预测值：{y_predict}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_predict)}')

    # 绘图
    plt.scatter(x, y)  # 以散点图绘制真实值
    # np.sort: 对x进行排序,默认升序
    # np.argsort: 对x进行排序，返回索引
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以折线图绘制预测值
    plt.show()

if __name__ == '__main__':
    #demo_under_fitting()
    #demo_just_fitting()
    demo_over_fitting()
    demo_l1_regularization()
    demo_l2_regularization()