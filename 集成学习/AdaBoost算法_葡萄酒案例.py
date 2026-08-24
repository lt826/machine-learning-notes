"""
集成学习-AdaBoost算法-葡萄酒案例

AdaBoost算法
    属于Boost思想，即串行执行，每次使用全部样本，最后加权投票
    原理：
        1.使用全部样本，通过决策树模型进行训练，获取结果
            思路：
                预测正确 ——> 权重下降
                预测错误 ——> 权重上升
        2.把第一个弱分类器的处理结果，交给第二个弱分类器，继续训练，获取结果
            思路：
                预测正确 ——> 权重下降
                预测错误 ——> 权重上升
        3.以此类推，串行执行，直至获取最终结果
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder          #标签编码器
from sklearn.model_selection import train_test_split    #训练集，测试集分割
from sklearn.tree import DecisionTreeClassifier         #决策树
from sklearn.ensemble import AdaBoostClassifier         #AdaBoost分类器
from sklearn.metrics import accuracy_score              #模型评估指标，计算模型预测的准确率

df = pd.read_csv("./data/wine.csv")
df = df.drop('Unnamed: 0', axis=1)  # 删除索引列

X = df[['alcohol', 'volatile acidity', 'sulphates']]  # 3个特征
y = (df['quality'] >= 5).astype(int)  # 标签：好酒(1) / 差酒(0)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)

#场景一：单一决策树，充当弱分类器
estimator1 = DecisionTreeClassifier(max_depth=3)
estimator1.fit(x_train, y_train)

y_pred1 = estimator1.predict(x_test)
print("单一决策树准确率：", accuracy_score(y_test, y_pred1))     #0.984375


#场景二：使用AdaBoost算法，集成学习，CART树
#参1：弱分类器，参2：弱分类器数量，参3：学习率
estimator2 = AdaBoostClassifier(estimator=estimator1, n_estimators=100,learning_rate=0.1)
estimator2.fit(x_train, y_train)
y_pred2 = estimator2.predict(x_test)
print("AdaBoost准确率：", accuracy_score(y_test, y_pred2))      #0.984375


