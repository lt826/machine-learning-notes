"""
案例：演示网格搜索和交叉验证

交叉验证：
    原理：
        把数据分成n份，例如4份：   --->也叫4折交叉验证
        第1次：把第1份作为测试集，其它为训练集，训练模型，模型预测，获取：准确率--> 准确率为1
        第2次：把第2份作为测试集，其它为训练集，训练模型，模型预测，获取：准确率--> 准确率为0.9
        第3次：把第3份作为测试集，其它为训练集，训练模型，模型预测，获取：准确率--> 准确率为0.8
        第4次：把第4份作为测试集，其它为训练集，训练模型，模型预测，获取：准确率--> 准确率为0.7
        最终准确率：(1+0.9+0.8+0.7)/4=0.85

        假如第4次最好，则：用全部数据训练模型，再次用（第4次的）测试集对模型测试

    目的：
        为了让模型的最终验证结果更准确

网格搜索：
    原理：
        接收超参可能出现的值，然后针对于超参的每个值进行交叉验证，获取到最优超参组合

    目的：
        寻找最优超参数，即模型的性能最佳的参数组合

网格搜索+交叉验证，本质上指的是GridSearchCV这个API，它会遍历指定的参数组合，并使用交叉验证评估每个参数组合的性能，最后选择表现最好的参数组合（供参考）。
"""

#导包
from sklearn.datasets import load_iris                                      #加载鸢尾花数据集
from sklearn.model_selection import train_test_split, GridSearchCV          #分割训练集和测试集，寻找最优超参数组合
from sklearn.preprocessing import StandardScaler                            #数据标准化
from sklearn.neighbors import KNeighborsClassifier                          #KNN算法 分类对象
from sklearn.metrics import accuracy_score                                  #模型评估指标，计算模型预测的准确率
import joblib

#1.加载数据
iris = load_iris()

#2.数据预处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=45)

#3.特征工程
#3.1特征预处理（标准化）
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#4.模型训练
#4.1创建KNN分类对象
estimator = KNeighborsClassifier()
#4.2超参可能出现的值
param_dict = {"n_neighbors": [i for i in range(1,11)]}
#4.3创建GridSearchCV对象
#参数1：要计算的模型对象，参数2：超参可能出现的值，参数3：折数，每个超参组合都会进行5折交叉验证，  这里是10*5=50次
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)
#4.4训练模型
estimator.fit(x_train, y_train)
#4.5打印最优超参数组合
print(f"最优评分：{estimator.best_score_}")          #
print(f"最优超参数组合：{estimator.best_params_}")    #'n_neighbors': 3
print(f"最佳估计器：{estimator.best_estimator_}")    #KNeighborsClassifier(n_neighbors=3)

#5.模型评估
#5.1获取最优超参的模型对象
best_estimator = estimator.best_estimator_        #等于KNeighborsClassifier(n_neighbors=3)
#5.2模型训练
best_estimator.fit(x_train, y_train)
#5.3模型预测
y_pre = best_estimator.predict(x_test)
#5.4模型评估
print(f"准确率为：{accuracy_score(y_test, y_pre)}")

#6.保存模型
joblib.dump(best_estimator, "./my_model/鸢尾花.pkl")           #pickle文件：Python（Pandas）独有的文件类型
print("模型保存成功！")





