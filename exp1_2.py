import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # 为了方便可视化，仅选取花瓣长度和花瓣宽度两个特征
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑斯蒂回归模型
model = LogisticRegression(max_iter=200)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# 可视化分类结果
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_test == y_pred][:, 0], X_test[y_test == y_pred][:, 1], color='green', marker='o', label='Correct', alpha=0.6)
plt.scatter(X_test[y_test != y_pred][:, 0], X_test[y_test != y_pred][:, 1], color='red', marker='x', label='Wrong', alpha=0.6)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Logistic Regression Classification Results')
plt.legend()
plt.show()
