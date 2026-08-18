"""
特征工程：
    利用专业的背景知识和技巧处理数据，让机器学习算法效果最好
    分为特征提取、特征预处理、特征降维、特征选择、特征组合

    归一化
        目的：
            防止因为量纲（单位）问题，导致特征列的方差值相差较大，影响模型的最终结果
        公式：
            X'=(当前值-该列最小值)/(该列最大值-该列最小值)
            X"=X' * (mx-mi) + mi

            X':基于公式算出来的结果
            X":最终的结果
            mx:区间的最大值
            mi:区间的最小值

        弊端：易受到最大值和最小值的影响，所以一般用于处理小数据集
"""

#1.导包
from sklearn.preprocessing import MinMaxScaler

#2.准备数据集
x_train = [[90,2,10],
           [60,4,15],
           [75,3,13]
           ]

#3.创建归一化对象
#参数feature_range：表示生成的范围，即mx和mi，默认为0-1
transfer = MinMaxScaler(feature_range=(3,7))

#4.对原数据集进行归一化操作
x_train_new = transfer.fit_transform(x_train)

#5.打印处理后的数据
print(f"归一化后的数据集为：{x_train_new}")