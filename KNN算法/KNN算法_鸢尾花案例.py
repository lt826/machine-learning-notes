"""
机器学习研发流程：
    1.数据采集      2.数据预处理      3.特征工程      4.模型训练      5.模型评估      6.模型预测
"""

#导包
from sklearn.datasets import load_iris                      #加载鸢尾花数据集
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split        #分割训练集和测试集
from sklearn.preprocessing import StandardScaler            #数据标准化
from sklearn.neighbors import KNeighborsClassifier          #KNN算法 分类对象
from sklearn.metrics import accuracy_score                  #模型评估指标，计算模型预测的准确率



#1.定义函数，加载鸢尾花数据集
def load_iris_data():
    iris_data = load_iris()
    #print(f"鸢尾花数据集为：{iris_data}")

#2.定义函数，绘制数据集的散点图
def plot_iris_data():
    #加载数据集
    iris_data = load_iris()
    #把鸢尾花数据集封装成DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    #print(f"鸢尾花数据集为：{iris_df}")
    #给df对象新增标签列
    iris_df['label'] = iris_data.target
    print(f"鸢尾花数据集为：{iris_df}")
    #通过Seaborn绘制散点图
    #参数1：x轴，参数2：y轴，参数3：分组字段，参数4：数据集
    sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', hue='label', data=iris_df)
    #设置标题
    plt.title('Iris Data')
    plt.tight_layout()      #自动调整子图参数，使之填充整个图像区域
    plt.show()

#3.定义函数，切分训练集和测试集
def split_train_test_data():
    iris_data = load_iris()
    #参数1：特征数据，参数2：标签数据，参数3：测试集占比，参数4：随机种子（种子一致，结果一致）
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)
    #打印切割后的结果
    print(f"训练集特征数据为：{X_train},个数为：{len(X_train)}")
    print(f"训练集标签数据为：{y_train},个数为：{len(y_train)}")
    print(f"测试集特征数据为：{X_test},个数为：{len(X_test)}")
    print(f"测试集标签数据为：{y_test},个数为：{len(y_test)}")

#4.定义函数，实现鸢尾花完整案例 --> 加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def iris_knn():

    #1.加载数据
    iris_data = load_iris()

    #2.数据预处理，把150条数据按照7:3的比例，切割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=18)

    #3.特征工程（提取，预处理…）
    #因为源数据只有4个特征，且都是我们需要用的，所以不需要进行特征提取
    #3.1特征预处理
    transfer = StandardScaler()
    #fit_transform：兼具fit和transform功能，即训练并转换数据，该函数适用于第一次进行标准化的时候使用，，一般处理：训练集
    x_train = transfer.fit_transform(X_train)
    #transform：转换数据，该函数适用于重复进行标准化动作时使用，一般处理：测试集
    x_test = transfer.transform(X_test)

    #4.模型训练
    #4.1 创建模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    #4.2 具体的训练模型的动作
    estimator.fit(x_train, y_train)         #传入训练集的特征数据和标签数据

    #5.模型预测
    #场景一：对刚才切分的测试集进行预测
    y_pre = estimator.predict(x_test)
    print(f"模型预测结果为：{y_pre}")

    #场景二：对新的数据进行预测
    my_data = [[5.1, 3.5, 1.4, 0.2]]
    #对新数据进行标准化
    my_data = transfer.transform(my_data)
    y_pre_new = estimator.predict(my_data)
    print(f"模型预测结果为：{y_pre_new}")

    #查看上述数据集，每种分类的预测概率
    y_pre_proba = estimator.predict_proba(my_data)
    print(f"(各分类)预测概率为：{y_pre_proba}")

    #6.模型评估
    #方式一：直接基于测试集的特征和标签进行评分
    print(f'正确率（准确率）：{estimator.score(x_test, y_test)}')

    #方式二：基于测试集的标签和预测结果进行评分
    print(f'正确率（准确率）：{accuracy_score(y_test, y_pre)}')


if __name__ == '__main__':
    #load_iris_data()
    #plot_iris_data()
    #split_train_test_data()
    iris_knn()