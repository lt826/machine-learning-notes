"""
随机森林算法
    1.属于bagging算法
    2.每个弱学习器都是CART树(必须是二叉树)
"""

import pandas as pd
from sklearn.model_selection import train_test_split        #分割训练集和测试集
from sklearn.tree import DecisionTreeClassifier             #CART树
from sklearn.ensemble import RandomForestClassifier         #随机森林算法
from sklearn.model_selection import GridSearchCV            #网格搜索

#1.加载数据
df = pd.read_csv("./data/titanic_train.csv")

#2.数据预处理
#2.1抽取特征和标签
x = df[['Pclass', 'Sex', 'Age']].copy()         #船舱等级，性别，年龄
y = df['Survived']

#2.2空值处理，用Age的平均值填充
x['Age'] = x['Age'].fillna(x['Age'].mean())        #用Age的平均值填充空值

#2.3热编码处理
x = pd.get_dummies(x)

#2.4划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

#3.特征工程，此处不需要

#4.模型训练
#场景一：创建单一决策树模型对象
estimator1 = DecisionTreeClassifier()
estimator1.fit(x_train, y_train)
y_pre1 = estimator1.predict(x_test)
print(f"决策树模型预测的准确率为：{estimator1.score(x_test, y_test)}")
print('-' * 20)

#场景二：创建随机森林模型对象
#4.1创建随机森林模型对象
estimator2 = RandomForestClassifier()       #n_estimators = 100,max_depth = None
estimator2.fit(x_train, y_train)
y_pre2 = estimator2.predict(x_test)
print(f"随机森林模型预测的准确率为：{estimator2.score(x_test, y_test)}")
print('-' * 20)

#场景三：随机森林算法 ——> 网格搜索
estimator3 = RandomForestClassifier()
estimator3.fit(x_train, y_train)
params = {
    'n_estimators': [30, 50, 70, 90, 110],
    'max_depth': [2, 3, 5, 7]
}
gs_estimator = GridSearchCV(estimator3, params, cv=2)
gs_estimator.fit(x_train, y_train)
print(f"网格搜索最佳参数为：{gs_estimator.best_params_}")  # ✅ 直接用 gs_estimator，GridSearchCV自动将最佳参数应用到模型中
print(f"随机森林模型(网格搜索)预测的准确率为：{gs_estimator.score(x_test,y_test)}")