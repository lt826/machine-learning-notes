"""
混淆矩阵：
    概述：
        用来描述真实值和预测值之间的关系的

    图解：
        True：真      False：假         Positive：正例         Negative：反例

                    预测值（正例）                 预测值（反例）
        真实值（正例）      真正例（TP）                  伪反例（FN）
        真实值（反例）      伪正例（FP）                  真反例（TN）

        真正例（TP）：预测为正例，实际也为正例
        假反例（FN）：预测为反例，实际为正例
        假正例（FP）：预测为正例，实际为反例
        真反例（TN）：预测为反例，实际也为反例

        精确率（Precision）：TP/(TP+FP)
        召回率（Recall）：TP/(TP+FN)
        F1分数（F1 Score）：2*精确率*召回率/(精确率+召回率)
"""

import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score            #混淆矩阵,精确率,召回率,F1值

#需求：已知有10个样本，6个恶性肿瘤（正例），4个良性肿瘤（反例）
#模型A预测结果为：预测对了3个恶性肿瘤，预测对了4个良性肿瘤
#模型B预测结果为：预测对了6个恶性肿瘤，预测对了1个良性肿瘤
#请针对上述的数据集，搭建混淆矩阵，并分别计算精确率、召回率和F1分数

#定义变量，记录样本数据
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

#定义变量，记录模型A的预测结果
y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']

#定义变量，记录模型B的预测结果
y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']

#用标签标记正例，反例
label = ['恶性', '良性']
df_label = ['恶性（正例）', '良性（反例）']

#针对于真实值和模型A的预测结果，搭建混淆矩阵
cm_A = confusion_matrix(y_train, y_pred_A, labels=label)
print("模型A的混淆矩阵为：\n", cm_A)

#美化，将混淆矩阵转换成DataFrame
df_A = pd.DataFrame(cm_A,index= df_label,columns= df_label)
print("模型A的混淆矩阵为：\n", df_A)

#针对于真实值和模型B的预测结果，搭建混淆矩阵
cm_B = confusion_matrix(y_train, y_pred_B, labels=label)
print("模型B的混淆矩阵为：\n", cm_B)

#美化，将混淆矩阵转换成DataFrame
df_B = pd.DataFrame(cm_B,index= df_label,columns= df_label)
print("模型B的混淆矩阵为：\n", df_B)


#计算模型A的精确率，召回率，F1值
print(f"模型A的精确率为：{precision_score(y_train, y_pred_A, pos_label='恶性')}")     #参1：真实值，参2：预测值，参3：正例标签
print(f"模型A的召回率为：{recall_score(y_train, y_pred_A, pos_label='恶性')}")
print(f"模型A的F1值为：{f1_score(y_train, y_pred_A, pos_label='恶性')}")

#计算模型B的精确率，召回率，F1值
print(f"模型B的精确率为：{precision_score(y_train, y_pred_B, pos_label='恶性')}")
print(f"模型B的召回率为：{recall_score(y_train, y_pred_B, pos_label='恶性')}")
print(f"模型B的F1值为：{f1_score(y_train, y_pred_B, pos_label='恶性')}")
