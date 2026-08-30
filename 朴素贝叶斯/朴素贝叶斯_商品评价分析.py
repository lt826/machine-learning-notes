"""
案例：通过朴素贝叶斯算法实现商品评价分析，即好评、差评

朴素贝叶斯：
    概述：
        贝叶斯：仅仅依赖概率就可以进行分类的一种机器学习算法
        朴素：不考虑特征之间的相关性，即特征之间是相互独立的
        原始： P(AB) = P(A) * P(B|A) = P(B) * P(A|B)
        加入朴素后： P(AB) = P(A) * P(B)

    细节：
        因为要分词所以用到jieba分词器
"""

import numpy as np
import pandas as pd
import jieba                                                # 分词器
import scipy.special.cython_special
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer #词频统计包，把评论内容转成词频矩阵
from sklearn.naive_bayes import MultinomialNB               # 朴素贝叶斯对象


#1.读取文件，获取数据
df = pd.read_csv("./data/书籍评价.csv",encoding='gbk')
#df.info()

#2.数据预处理
#2.1添加labels列，充当标签列。  1表示好评，0表示差评
df['labels'] = np.where(df['评价']=='好评',1,0)
#print(df)

#2.2抽取labels列作为标签
y = df['labels']

#2.3演示jieba分词
#print(jieba.lcut("我爱自然语言处理,一把把把把住了！轻舟已过万重山"))

#2.4对用户评论进行分词
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]

#2.5加载停用词列表，停用词列表中包含的词在分词时会被过滤掉
with open('./data/stopwords.txt','r',encoding='utf-8') as f:
    #一次读取所以行
    stopwords = f.readlines()
    #删除最后的\n
    stopwords = [line.strip() for line in stopwords]
    #对停用词去重
    stopwords = list(set(stopwords))
    print(stopwords)

#2.6创建向量化对象，从评论词列表（comment_list）中删除停用词，并统计词频（单词矩阵）
transfer = CountVectorizer(stop_words=stopwords)

#2.7统计词频矩阵
transfer.fit(comment_list)
#x的格式：[[第一条评论的切词分布，有就是1，没有就是0],[第二条评论的切词分布，有就是1，没有就是0]...]
x = transfer.transform(comment_list).toarray()

#2.8切分数据集和测试集
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

#3.模型训练
estimator = MultinomialNB()
estimator.fit(x_train,y_train)

#4.模型预测
y_pre = estimator.predict(x_test)

#5.模型评估
print(f"准确率为：{accuracy_score(y_test, y_pre)}")
