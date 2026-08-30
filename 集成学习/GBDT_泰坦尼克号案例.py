"""
GBDT(Gradient Boosting Decision Tree)梯度提升树：
    通过拟合负梯度来获取一个强学习器
    流程：
        1.采用所有目标值的均值作为初始预测值
        2.负梯度(残差) = 目标值 - 预测值,该值作为新的目标值，进行下一次预测
        3.针对于第1个学习器，依次计算每个分割点的最小平方和，找到最佳分割点
        4.将上述分割点带入第二个学习器，计算它的预测值 = 以此分割点为界，目标值的均值，即为该部分数据的预测值
        5.计算第二个学习器的负梯度，最佳分割点
        6.重复上述过程，直至达到预设的迭代次数
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier                 #决策树
from sklearn.ensemble import GradientBoostingClassifier         #梯度提升树
from sklearn.model_selection import GridSearchCV                #网格搜索
from sklearn.metrics import accuracy_score                      #模型评估指标，计算模型预测的准确率

df = pd.read_csv("./data/titanic_train.csv")
x = df[['Pclass', 'Sex', 'Age']].copy()         #船舱等级，性别，年龄
y = df['Survived']

x['Age'] = x['Age'].fillna(x['Age'].mean())        #用Age的平均值填充空值

x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

#场景一：单个决策树
estimator1 = DecisionTreeClassifier()
estimator1.fit(x_train,y_train)
y_pre1 = estimator1.predict(x_test)
#print(f'单个决策树预测的分类报告：\n{classification_report(y_test, y_pre1)}')
print(f"单个决策树模型预测的准确率为：{accuracy_score(y_test, y_pre1)}")
print('-' * 20)

#场景二：GBDT梯度提升树
estimator2 = GradientBoostingClassifier()
estimator2.fit(x_train,y_train)
y_pre2 = estimator2.predict(x_test)
#print(f'单个梯度提升树预测的分类报告：\n{classification_report(y_test, y_pre2)}')
print(f"单个梯度提升树模型预测的准确率为：{accuracy_score(y_test, y_pre2)}")

print('-' * 20)

#场景三：基于GBDT模型，进行参数调优
param_dict = {
    'n_estimators': [50, 100, 130],          #弱学习器数量
    'learning_rate': [0.01, 0.1, 0.2],      #学习率
    'max_depth': [3, 5, 7]                  #树的最大深度
}

estimator3 = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_dict, cv=3)
estimator3.fit(x_train, y_train)
print(f"最佳模型为：{estimator3.best_estimator_}")
y_pre3 = estimator3.predict(x_test)
print(f"基于GBDT模型，进行参数调优后的准确率为：{accuracy_score(y_test, y_pre3)}")

