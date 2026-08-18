"""
特征工程：
    利用专业的背景知识和技巧处理数据，让机器学习算法效果最好
    分为特征提取、特征预处理、特征降维、特征选择、特征组合

    标准化：通过对原数据进行标准化，转为均值为0标准差为1的标准正态分布的数据
        目的：
            防止因为量纲（单位）问题，导致特征列的方差值相差较大，影响模型的最终结果
            通过公式把各列的值映射到均值为0标准差为1的正态分布序列
        公式：
            X'=(当前值-该列平均值) / 该列的标准差

        应用场景：
            用于处理大数据集
"""

#1.导包
from sklearn.preprocessing import StandardScaler

#2.准备数据集
x_train = [[90,2,10],
           [60,4,15],
           [75,3,13]
           ]

#3.创建标准化对象
#参数feature_range：表示生成的范围，即mx和mi，默认为0-1
transfer = StandardScaler()

#4.对原数据集进行标准化操作
x_train_new = transfer.fit_transform(x_train)

#5.打印处理后的数据
print(f"标准化后的数据集为：{x_train_new}")

#6.打印数据集的均值和方差
print(f"数据集的均值为：{transfer.mean_}，数据集的方差为：{transfer.var_}，数据集的标准差为：{transfer.scale_}")