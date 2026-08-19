#导包
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression         #逻辑回归模型
from sklearn.preprocessing import StandardScaler            #标准化
from sklearn.model_selection import train_test_split        #训练集和测试集分离
from sklearn.metrics import accuracy_score                  #模型评估

#1.加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
#data.info()        #查看信息

#2.数据预处理
#2.1 把？替换成np.nan     参1：要替换的值     参2：替换后的值     参3：是否替换源数据，默认为False
data.replace('?',np.nan,inplace=True)
#2.2缺失值处理  --->  填值或删除，这里选择删除
data.dropna(axis= 0 ,inplace=True)      #axis=0表示按行删除，inplace=True表示替换原数据

#3.特征工程
#3.1特征提取
X = data.iloc[:,1:-1]           #按照行号，列索引获取数据
y = data.iloc[:,-1]             #获取最后一列
#3.2切割训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)
#3.3标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test= transfer.transform(x_test)

#4.模型训练
#创建逻辑回归模型对象
estimator = LogisticRegression()
#4.1模型训练
estimator.fit(x_train,y_train)

#5.模型预测
y_pre = estimator.predict(x_test)
print(f'预测值为：{y_pre}')

#6.模型评估
#正确率（准确率），公式为：预测正确的/总样本数
print(f"预测前评估，正确率为{estimator.score(x_test,y_test)}")        #测试集的特征和标签
print(f"预测前评估，正确率为{accuracy_score(y_test, y_pre)}")        #测试集的标签和预测值