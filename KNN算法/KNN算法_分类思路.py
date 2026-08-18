"""
KNN算法介绍（K Nearest Neighbors），K近邻算法
    原理：
        基于欧式距离（或其它距离计算方法）计算测试集和每个训练集的距离，然后根据距离升序排列，找到最近的K个样本
        基于K个样本投票，票数多的就作为最终预测结果 ————> 分类问题
"""

#1.导包
from sklearn.neighbors import KNeighborsClassifier

#2.准备测试集和训练集
x_train = [[9, 3], [2, 4], [2, 5], [5, 6]]     #训练集的特征数据，特征可以有多个，所以是二维数组
y_train = [0,1,0,1]                            #训练集的标签数据
x_test = [[9,1]]                               #测试集的特征数据

#3.创建KNN模型对象
estimator = KNeighborsClassifier(n_neighbors=2)

#4.模型训练
estimator.fit(x_train,y_train)

#5.模型预测
y_pre = estimator.predict(x_test)

#6.打印预测结果
print(f"预测结果为{y_pre}")
