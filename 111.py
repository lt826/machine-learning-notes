import numpy as np
from sklearn.linear_model import LinearRegression

# 训练数据：面积（平米）和对应的房价（万元）
X = np.array([[50], [70], [90], [110], [130]])  # 特征（必须二维）
y = np.array([150, 210, 280, 350, 420])         # 标签

# 创建模型并训练
model = LinearRegression()
model.fit(X, y)

# 预测 100 平米的房价
pred = model.predict([[20]])
print(f"100平米的预测房价：{pred[0]:.2f} 万元")