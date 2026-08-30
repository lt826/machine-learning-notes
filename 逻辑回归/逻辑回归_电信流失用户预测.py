"""
案例：
    通过逻辑回归算法，针对于电信用户数据建模，进行用户流失预测
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report    #准确率，精确率，召回率，F1分数，分类评估报告

#数据预处理
def demo01_data_preprocess():
    #1.读取csv文件，获取到df对象
    churn_df = pd.read_csv('./data/churn.csv')
    #2.查看（处理前）的数据集
    #churn_df.info()
    #print(churn_df.head(5))
    #3.因为Churn，gender列是字符串，所以需要进行one-hot编码（热编码），能将Churn，gender列转换为Churn_No，Churn_Yes，gender_Male，gender_Female四列
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    #4.查看（处理后）的数据集
    #churn_df.info()
    #print(churn_df.head(5))
    #5.删除one-hot处理后冗余的列
    #参1：要删除的列名，参2：axis=1表示删除列，参3：inplace=True表示直接在原数据上进行删除
    churn_df.drop(['Churn_No', 'gender_Female'], axis=1, inplace=True)
    churn_df.info()
    print(churn_df.head(5))
    #6.修改列名，将Churn_Yes --> flag,充当标签列
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    churn_df.info()
    print(churn_df.head(5))         #False ---> 不流失，True ---> 流失
    #7.查看数据值的分布
    print(churn_df['flag'].value_counts())


#数据可视化
def demo02_data_visualization():
    churn_df = pd.read_csv('./data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Female'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    #数据可视化，绘制计数柱状图
    #参1：数据集，参2：x轴的列名（月度会员），参3：分组字段
    sns.countplot(data=churn_df, x='Contract_Month', hue='flag')
    plt.show()


#模型训练，预测，评估
def demo03_model_train_predict_evaluate():
    churn_df = pd.read_csv('./data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Female'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    x = churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    y = churn_df['flag']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

    estimator = LogisticRegression()        #逻辑回归模型
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(f'预测值为:{y_predict}')

    print(f'准确率为:{accuracy_score(y_test, y_predict)}')
    print(f'精确率为:{precision_score(y_test, y_predict)}')
    print(f'召回率为:{recall_score(y_test, y_predict)}')
    print(f'F1分数为:{f1_score(y_test, y_predict)}')
    #macro avg:宏平均，即不考虑样本权重，直接求平均，适用于：数据均衡的情况
    #weighted avg:样本权重平均，即考虑样本权重，求平均，适用于：数据不均衡的情况
    print(f'分类评估报告为:\n{classification_report(y_test, y_predict)}')

if __name__ == '__main__':
    #demo01_data_preprocess()
    #demo02_data_visualization()
    demo03_model_train_predict_evaluate()