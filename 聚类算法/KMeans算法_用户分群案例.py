"""
案例：
    基于用户的年收入和消费质数，根据用户的相似性进行聚类
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'         #设置OMP程序运行时使用的线程数

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

#定义函数，找聚类的质心树（K值）
def dm01_find_k():
    df = pd.read_csv('./data/customers.csv')

    #定义sse_list,sc_list，记录不同k值的评估效果
    sse_list = []
    sc_list = []

    #抽取特征
    x = df.iloc[:,3:5]

    #for循环，测试不同k值的评估效果
    for k in range(2,20):
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=12)
        estimator.fit(x)
        y_pre = estimator.predict(x)
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x, y_pre))

    #绘图
    plt.figure(figsize=(12, 6))
    plt.plot(range(2,20), sse_list, label='SSE')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(2,20), sc_list, label='SC')
    plt.show()

#定义函数，实现：模型训练，模型预测，模型评估
def dm02_kmeans_cluster():
    df = pd.read_csv('./data/customers.csv')
    x = df.iloc[:,3:5]
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=23)
    estimator.fit(x)
    y_pre = estimator.predict(x)

    #绘制5个簇的样本点 --> 散点图
    plt.scatter(x.values[y_pre == 0,0], x.values[y_pre == 0,1],label = 'Standard')          #0号簇
    plt.scatter(x.values[y_pre == 1,0], x.values[y_pre == 1,1],label = 'Traditional')       #1号簇
    plt.scatter(x.values[y_pre == 2,0], x.values[y_pre == 2,1],label = 'Normal')            #2号簇
    plt.scatter(x.values[y_pre == 3,0], x.values[y_pre == 3,1],label = 'Youth')             #3号簇
    plt.scatter(x.values[y_pre == 4,0], x.values[y_pre == 4,1],label = 'TA')                #4号簇

    #绘制5个簇的质心点 --> 散点图
    plt.scatter(estimator.cluster_centers_[:,0], estimator.cluster_centers_[:,1],label='Center')

    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #dm01_find_k()
    dm02_kmeans_cluster()
