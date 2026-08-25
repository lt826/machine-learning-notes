"""
XGBoost(极限梯度提升树，EXtreme Gradient Boosting Tree)
    基于打分函数的结果决定是否分枝
    打分函数：
        obj = -1/2 * ∑(G² / (H²+λ)) + γT
        G：所有叶节点的一阶导
        H：所有叶节点的二阶导
        λ，γ：参数
        T：叶节点个数

    Gain值
        分支前的打分函数 - 分支后左子树的打分函数 - 分支后右子树的打分函数
        Gain = obj(parent) - (obj(left) + obj(right))

        如果Gain > 0, 则分裂之后树的分数比分裂之前更小，会考虑此次分裂
        如果Gain < 0, 则分裂之后树的分数比分裂之前更大，不会考虑此次分裂
"""

import joblib                                           #保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb                                   #极限梯度提升树对象
from collections import Counter                         #统计数据
from sklearn.model_selection import train_test_split    #训练集和测试集分割
from sklearn.metrics import accuracy_score              #准确率
from sklearn.metrics import classification_report       #分类报告
from sklearn.model_selection import StratifiedKFold     #分层K折交叉验证
from sklearn.utils import class_weight                  #计算样本权重
from sklearn.model_selection import GridSearchCV        #网格搜索

#定义函数，对红酒数据进行分割，并存储到csv文件中
def dm01_data_split():
    df = pd.read_csv("./data/wine.csv")
    df = df.drop('Unnamed: 0', axis=1)  # 删除索引列

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    #print(f"查看标签结果的分布情况：{Counter(y)}")

    #参5：参考数据集的标签分布
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

    #把上述的训练集特征和标签数据拼接到一起，测试集特征和标签数据拼接到一起，写入文件中
    #axis=1: 按列拼接, index=False:忽略索引
    pd.concat([x_train, y_train], axis=1).to_csv("./data/wine_train.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv("./data/wine_test.csv", index=False)


#定义函数，训练模型并保存
def dm02_train_model():
    df_train = pd.read_csv("./data/wine_train.csv")
    df_test = pd.read_csv("./data/wine_test.csv")
    x_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    x_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    #创建模型对象
    estimaor = xgb.XGBClassifier(
        max_depth=5,        #树的最大深度
        learning_rate=0.1,  #学习率
        n_estimators=100,   #树的个数
        random_state=23,    #随机种子
        objective='multi:softmax',  #多分类问题，使用多分类模型
    )

    estimaor.fit(x_train, y_train)
    print(f"模型的准确率为：{estimaor.score(x_test, y_test)}")

    #保存模型
    joblib.dump(estimaor, './model/红酒分类模型.pkl')
    print("模型保存成功！")


#定义函数，训练模型并保存
def dm03_use_model():
    df_train = pd.read_csv("./data/wine_train.csv")
    df_test = pd.read_csv("./data/wine_test.csv")
    x_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    x_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    #加载模型
    estimator = joblib.load('./model/红酒分类模型.pkl')

    #创建网格搜索 + 交叉验证（结合分层采样数据），找模型最优参数组合
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [50, 100, 120],
    }

    #创建采样对象
    #参1：折数，参2：是否打乱数据，参3：随机种子
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

    #参1：模型对象，参2：参数组合，参3：交叉验证对象
    gs_estimator = GridSearchCV(estimator, param_grid, cv=skf)

    gs_estimator.fit(x_train, y_train)

    y_pre = gs_estimator.predict(x_test)

    print(f'最优参数: {gs_estimator.best_params_}')
    print(f'最优分数: {gs_estimator.best_score_}')
    print(f'准确率: {accuracy_score(y_test, y_pre)}')

if __name__ == '__main__':
    #dm01_data_split()
    #dm02_train_model()
    dm03_use_model()