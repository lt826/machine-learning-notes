"""
演示聚类算法评估指标，包括SC轮廓系数法、SSE + 肘部法、CH轮廓系数法
    聚类算法评估指标：
        1.SSE + 肘部法
            SSE：
                所有簇的所有样本到该簇中心的误差的平方和
                越小，表示簇内样本越聚集，内聚程度越高
                随着K值增加，SSE值会逐渐减小

            肘部法：
                K值增加，SSE值会逐渐减小，下降梯度陡然变缓

        2.SC轮廓系数法
            考虑簇内 ---> 聚集程度，越小越好
            考虑簇外 ---> 分离程度，越大越好

        3.CH轮廓系数法
            考虑簇内 ---> 聚集程度，越大越好
            考虑簇外 ---> 分离程度，越大越好
            考虑K值 ---> K值越小，代表簇内样本越聚集，内聚程度越高

"""
import os
os.environ['OMP_NUM_THREADS'] = '4'         #设置OMP程序运行时使用的线程数

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

#定义函数，演示SSE + 肘部法
def dm01_sse():
    #1.定义sse列表，记录每个K值的SSE
    sse_list = []

    #2.生成数据集
    x , y =make_blobs(
        n_samples=1000,                     #样本数量
        n_features=2,                       #特征数量
        centers=[[0,0],[1,1],[2,2],[3,3]],  #4个簇
        cluster_std=[0.4,0.2,0.3,0.3],      #4个簇的标准差
        random_state=12                     #随机种子
    )

    #3.for循环遍历，获取到每个K值，计算对应的sse值，并添加到sse列表中
    for k in range(1,101):
        #3.1创建KMeans对象，参1：聚类数量，参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=12)
        #3.2模型训练
        estimator.fit(x)
        #3.3获取sse值，
        sse_value = estimator.inertia_
        #3.4加到sse列表中
        sse_list.append(sse_value)

    #4.绘制图形
    #4.1创建画布
    plt.figure(figsize=(20,10))
    #4.2设置标题
    plt.title('sse value')
    #4.3设置x轴刻度
    plt.xticks(range(1,101,3))
    #4.4添加x轴，y轴标签
    plt.xlabel('k')
    plt.ylabel('sse')
    #4.5绘制网格
    plt.grid()
    #4.6绘制折线图
    #参1：K值，参2：sse值
    plt.plot(range(1,101), sse_list)
    plt.show()


#定义函数，演示SC轮廓系数法
def dm02_sc():
    # 1.定义sc列表，记录每个K值的sc
    sc_list = []

    # 2.生成数据集
    x, y = make_blobs(
        n_samples=1000,  # 样本数量
        n_features=2,  # 特征数量
        centers=[[0, 0], [1, 1], [2, 2], [3, 3]],  # 4个簇
        cluster_std=[0.4, 0.2, 0.3, 0.3],  # 4个簇的标准差
        random_state=12  # 随机种子
    )

    # 3.for循环遍历，获取到每个K值，计算对应的sc值，并添加到sc列表中
    for k in range(2, 101):         #考虑簇外，至少需要2个簇
        # 3.1创建KMeans对象，参1：聚类数量，参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=12)
        # 3.2模型训练和预测
        estimator.fit(x)
        y_pred = estimator.predict(x)
        # 3.3获取sc值，
        sc_value = silhouette_score(x, y_pred)
        # 3.4加到sc列表中
        sc_list.append(sc_value)

    # 4.绘制图形
    # 4.1创建画布
    plt.figure(figsize=(20, 10))
    # 4.2设置标题
    plt.title('sc value')
    # 4.3设置x轴刻度
    plt.xticks(range(1, 101, 3))
    # 4.4添加x轴，y轴标签
    plt.xlabel('k')
    plt.ylabel('sc')
    # 4.5绘制网格
    plt.grid()
    # 4.6绘制折线图
    # 参1：K值，参2：sc值
    plt.plot(range(2, 101), sc_list)
    plt.show()

#定义函数，演示CH轮廓系数法
def dm03_ch():
    # 1.定义ch列表，记录每个K值的ch
    ch_list = []

    # 2.生成数据集
    x, y = make_blobs(
        n_samples=1000,  # 样本数量
        n_features=2,  # 特征数量
        centers=[[0, 0], [1, 1], [2, 2], [3, 3]],  # 4个簇
        cluster_std=[0.4, 0.2, 0.3, 0.3],  # 4个簇的标准差
        random_state=12  # 随机种子
    )

    # 3.for循环遍历，获取到每个K值，计算对应的ch值，并添加到ch列表中
    for k in range(2, 101):         #考虑簇外，至少需要2个簇
        # 3.1创建KMeans对象，参1：聚类数量，参2：最大迭代次数，参3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=12)
        # 3.2模型训练和预测
        estimator.fit(x)
        y_pred = estimator.predict(x)
        # 3.3获取ch值，
        ch_value = calinski_harabasz_score(x, y_pred)
        # 3.4加到ch列表中
        ch_list.append(ch_value)

    # 4.绘制图形
    # 4.1创建画布
    plt.figure(figsize=(20, 10))
    # 4.2设置标题
    plt.title('ch value')
    # 4.3设置x轴刻度
    plt.xticks(range(1, 101, 3))
    # 4.4添加x轴，y轴标签
    plt.xlabel('k')
    plt.ylabel('ch')
    # 4.5绘制网格
    plt.grid()
    # 4.6绘制折线图
    # 参1：K值，参2：ch值
    plt.plot(range(2, 101), ch_list)
    plt.show()

if __name__ == '__main__':
    #dm01_sse()
    #dm02_sc()
    dm03_ch()
