"""
KMeans:
    属于无监督学习，即有特征，无标签
    所谓的相似性可以理解为距离，距离越小，相似度越高，例如欧式距离、曼哈顿距离、切比雪夫距离

"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans                      #KMeans算法
from sklearn.datasets import  make_blobs                #默认按照正态分布生成数据集，只需指定的均值和标准差
from sklearn.metrics import calinski_harabasz_score     #评价指标，值越大，表示聚类效果越好


#1.准备数据集
#参1：样本数量，参2：样本特征数量（2列），参3：样本标签数量，参4 标准差，参5：随机种子
x,y = make_blobs(n_samples=1000, n_features=2, centers=[[0,0],[1,1],[2,2],[3,3]], cluster_std=[0.4,0.2,0.3,0.3], random_state=35)

#2.绘制上述图形
#参1：横坐标，参2：纵坐标
#plt.scatter(x[:,0],x[:,1])
#plt.show()

#3.创建KMeans对象
#参1：聚类数量(簇)，参2：随机种子
estimator = KMeans(n_clusters=4,random_state=35)

#4.模型训练和预测
y_pre = estimator.fit_predict(x)

plt.scatter(x[:,0],x[:,1],c= y_pre)
plt.show()

#5.评价指标
print(f"评价指标为:{calinski_harabasz_score(x, y_pre)}")


