# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()

# 选取BMI特征（在数据集中的第三列）
X_bmi = diabetes.data[:, np.newaxis, 2]

#打印bmi
print(X_bmi)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_bmi, diabetes.target, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 绘制结果
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear regression')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Linear Regression on Diabetes Dataset')
plt.legend()
plt.show()
