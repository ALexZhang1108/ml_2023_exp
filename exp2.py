# 导入必要的库
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 生成数据集
X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('数据集可视化')
plt.show()

# 构建模型
model = LinearSVC(C=1.0)  # 默认参数C=1.0

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确度: {accuracy:.2f}")

# 分析讨论
# 你可以尝试不同的C值，例如C=0.1, C=10, C=100 等，观察模型在测试集上的准确度如何随C的改变而变化
