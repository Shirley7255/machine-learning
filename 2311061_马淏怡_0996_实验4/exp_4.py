import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize  # 用于多分类 ROC/AUC
from itertools import cycle

# =================================================================
# 1. 数据加载与预处理 (semeion.data) - 基础任务
# =================================================================

try:
    # 确保文件名为 'semeion.data'
    raw_data = np.loadtxt('semeion.data')
except FileNotFoundError:
    print("错误：未找到 'semeion.data' 文件。")
    exit()

X = raw_data[:, :256]
y_onehot = raw_data[:, 256:]
y = np.argmax(y_onehot, axis=1)

# **分层采样实现 (7:3 划分)**
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"原始数据集样本数: {len(X)}")
print(f"训练集样本数 (70%): {len(X_train)}")
print(f"测试集样本数 (30%): {len(X_test)}")


# =================================================================
# 2. 自实现朴素贝叶斯分类器 (高斯朴素贝叶斯) - 基础任务 (已增强 predict_proba)
# =================================================================

class NaiveBayesClassifier:
    """自实现高斯朴素贝叶斯分类器 (Gaussian Naive Bayes)"""

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            # 添加平滑项 (1e-6) 避免方差为零
            self._var[idx, :] = X_c.var(axis=0) + 1e-6
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        """高斯概率密度函数 (PDF) 计算 P(x_i | C_k)，包含数值平滑"""
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator

        # 关键修正：确保概率值不会为 0，避免 log(0) 警告
        return np.maximum(prob, 1e-9)

    def _calculate_probabilities(self, x):
        """计算单个样本 x 属于所有类别的后验概率，并进行归一化"""
        log_posteriors = []
        for idx in range(len(self._classes)):
            log_prior = np.log(self._priors[idx])
            # np.sum(np.log(self._pdf(idx, x))) 避免 log(0)
            log_likelihood = np.sum(np.log(self._pdf(idx, x)))
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)

        # 将对数后验概率转换为实际概率，并进行归一化 (Softmax)
        exp_posteriors = np.exp(log_posteriors)
        return exp_posteriors / np.sum(exp_posteriors)

    def predict_proba(self, X):
        """输出每个样本属于每个类别的概率 (高级任务必需)"""
        # 确保输入是 numpy 数组
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        return np.array([self._calculate_probabilities(x) for x in X_np])

    def predict(self, X):
        """对测试集进行批量预测"""
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        y_pred = [self._classes[np.argmax(self._calculate_probabilities(x))] for x in X_np]
        return np.array(y_pred)


# 实例化并训练分类器
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)

# =================================================================
# 3. 中级任务：分类模型多维度评估
# =================================================================

print("\n" + "=" * 60)
print("中级任务：分类模型多维度评估")
print("=" * 60)

conf_matrix = confusion_matrix(y_test, y_pred)
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, y_pred, average='macro', zero_division=0
)
precision_all, recall_all, fscore_all, support_all = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=classifier._classes, zero_division=0
)

print("\n### 1. 混淆矩阵###")
print(conf_matrix)

print("\n### 2. 宏平均评估指标 ###")
print(f"总准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"宏平均精度: {precision:.4f}")
print(f"宏平均召回率: {recall:.4f}")
print(f"宏平均 F1-Score (F值): {fscore:.4f}")

# 格式化输出各类别详细评估指标
metrics_df = pd.DataFrame({
    '类别': classifier._classes,
    '支持样本数': support_all,
    '精度 (P)': [f'{p:.4f}' for p in precision_all],
    '召回率 (R)': [f'{r:.4f}' for r in recall_all],
    'F1-Score (F)': [f'{f:.4f}' for f in fscore_all]
})
print("\n### 3. 各类别详细评估指标 ###")
try:
    print(metrics_df.to_markdown(index=False))
except ImportError:
    print("缺少 tabulate 库，无法格式化输出。详细指标如下：")
    print(metrics_df)

# =================================================================
# 4. 高级任务：ROC 曲线与 AUC 计算和绘图
# =================================================================

print("\n" + "=" * 60)
print("高级任务：ROC 曲线与 AUC 计算")
print("=" * 60)

# 1. 获取模型预测概率
y_score = classifier.predict_proba(X_test)

# 2. 将真实标签 y_test 转换为独热编码 (One-Hot Encoding)
classes = classifier._classes
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)

# 3. 计算每个类别的 ROC 曲线和 AUC (OvR 策略)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 4. 计算微平均 (Micro-average) ROC 曲线和 AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 5. 绘制所有类别的 ROC 曲线
plt.figure(figsize=(10, 8))

# 绘制微平均 ROC 曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# 绘制每个类别的 ROC 曲线
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.4f})'
                   ''.format(classes[i], roc_auc[i]))

# 绘制对角线 (随机分类器)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Recall')
plt.title('ROC Curve for Multi-class Naive Bayes Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 6. 打印 AUC 结果
print("\n### AUC 值 (Area Under Curve) ###")
print(f"微平均 AUC (Micro-average AUC): {roc_auc['micro']:.4f}")
print("各类别 AUC 值:")
for i in range(n_classes):
    print(f"  类别 {classes[i]}: {roc_auc[i]:.4f}")

