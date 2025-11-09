"""
实验四：贝叶斯分类器

实验目标：
在数据集上应用贝叶斯规则进行分类，计算分类错误率，分析实验结果

数据已生成，请自行实现贝叶斯分类算法
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import train_test_split

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# TODO: 请在此处实现贝叶斯分类器
# ============================================================
class MyBayesianClassifier:
    """
    高斯朴素贝叶斯分类器
    使用最大后验概率规则 (MAP) 进行决策
    """

    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None  # P(C_i)
        self.class_means_ = None  # 均值 mu_i
        self.class_vars_ = None  # 方差 sigma^2_i
        self.n_features_ = 0

    def fit(self, X, y):
        """
        训练模型：估计每个类别的均值、方差和先验概率
        """
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]

        self.class_priors_ = {}
        self.class_means_ = {}
        self.class_vars_ = {}

        for c in self.classes_:
            X_c = X[y == c]

            # 估计先验概率 P(C_i)
            self.class_priors_[c] = len(X_c) / n_samples

            # 估计均值 mu_i (沿特征轴求平均)
            self.class_means_[c] = np.mean(X_c, axis=0)

            # 估计方差 sigma^2_i (沿特征轴求方差)
            # 加上一个微小值 (如 1e-6) 防止方差为零
            self.class_vars_[c] = np.var(X_c, axis=0) + 1e-6

    def _calculate_log_likelihood(self, X):
        """
        计算对数似然 log P(X|C_i)，利用朴素贝叶斯假设
        公式: log P(X|C_i) = sum_j [ -0.5 * log(2*pi*sigma^2_ij) - (x_j - mu_ij)^2 / (2*sigma^2_ij) ]
        """
        n_samples = X.shape[0]
        log_likelihoods = np.zeros((n_samples, len(self.classes_)))

        for i, c in enumerate(self.classes_):
            mean = self.class_means_[c]
            var = self.class_vars_[c]

            # 1. 常数项和方差项: -0.5 * log(2*pi*sigma^2)
            log_det = -0.5 * np.sum(np.log(2. * np.pi * var))

            # 2. 距离项: - (x - mu)^2 / (2*sigma^2)
            exponent = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

            # 3. 对数似然 = 常数项 + 距离项
            log_likelihoods[:, i] = log_det + exponent

        return log_likelihoods

    def predict(self, X):
        """
        最大后验概率规则 (MAP) 进行分类
        决策规则: C* = argmax [log P(X|C_i) + log P(C_i)]
        """
        # 1. 计算对数似然 log P(X|C_i)
        log_likelihoods = self._calculate_log_likelihood(X)

        # 2. 计算对数先验 log P(C_i)
        log_priors = np.array([np.log(self.class_priors_[c]) for c in self.classes_])

        # 3. 计算对数后验（对数联合概率）
        # NumPy广播机制自动完成 (n_samples, n_classes) + (n_classes,)
        log_joint = log_likelihoods + log_priors

        # 4. 选择对数后验概率最大的类别索引
        predictions_indices = np.argmax(log_joint, axis=1)

        # 5. 将索引映射回类别标签
        return np.array([self.classes_[i] for i in predictions_indices])


# ============================================================
# 辅助函数：绘制决策边界
# ============================================================

def plot_decision_boundary(model, X, y, title, filename):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格点的类别
    # 注意：这里我们使用整个数据集 X 来定义网格范围，但使用训练好的模型来预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制决策区域
    cmap_boundary = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, cmap=cmap_boundary, alpha=0.6)

    # 绘制数据点
    cmap_data = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_data,
               edgecolor='k', s=20, label='测试数据')

    # 计算测试集错误率
    X_test_plot = X  # 假设这里使用测试集X_test
    y_test_plot = y  # 假设这里使用测试集y_test

    predictions = model.predict(X_test_plot, )
    error_rate = 1 - accuracy_score(y_test_plot, predictions)

    ax.set_title(f'{title}\n贝叶斯分类 (MAP) 错误率={error_rate:.4f}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()

    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


# ============================================================
# 运行实验
# ============================================================

def run_experiment_four(X_train, X_test, y_train, y_test, dataset_name, output_dir):
    """运行实验四"""
    print(f'\n{"=" * 60}')
    print(f'{dataset_name}')
    print(f'{"=" * 60}')

    # 步骤1: 训练模型
    print('[步骤1] 训练贝叶斯分类器...')
    model = MyBayesianClassifier()
    model.fit(X_train, y_train)
    print('  模型训练完成。')

    # 步骤2: 在测试集上预测并评估
    print('[步骤2] 在测试集上评估...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    print(f'  测试集样本数: {len(X_test)}')
    print(f'  分类错误率: {error_rate:.4f}')

    # 步骤3: 可视化 (使用测试数据点和训练得到的决策边界)
    print('[步骤3] 生成决策边界图...')
    safe_name = dataset_name.replace(' ', '_')
    plot_decision_boundary(model, X_test, y_test, dataset_name,
                           os.path.join(output_dir, f'{safe_name}_Bayesian.png'))

    return error_rate

if __name__ == '__main__':
    output_dir = 'out'
    # 创建输出目录
    os.makedirs('out', exist_ok=True)

    # ============================================================
    # 数据生成
    # ============================================================

    print('=' * 60)
    print('实验四：贝叶斯分类器')
    print('=' * 60)

    # 生成数据集1：高分离度
    print('\n生成数据集1 (高分离度)...')
    X1, y1 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )

    # 生成数据集2：低分离度
    print('生成数据集2 (低分离度)...')
    X2, y2 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=0.5,
        random_state=42
    )

    # 划分训练集和测试集
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

    print(f'\n数据集1: 训练集{X1_train.shape[0]}样本, 测试集{X1_test.shape[0]}样本')
    print(f'数据集2: 训练集{X2_train.shape[0]}样本, 测试集{X2_test.shape[0]}样本')

    # ============================================================
    # 示例：可视化数据分布
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 数据集1
    axes[0].scatter(X1[y1 == 0, 0], X1[y1 == 0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
    axes[0].scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    axes[0].set_title('数据集1 (高分离度)')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 数据集2
    axes[1].scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
    axes[1].scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    axes[1].set_title('数据集2 (低分离度)')
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('out/实验四_数据分布.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('\n数据分布图已保存: out/实验四_数据分布.png')

    # 运行实验四
    error1 = run_experiment_four(X1_train, X1_test, y1_train, y1_test,
                                 '数据集1 (高分离度)', 'out')
    error2 = run_experiment_four(X2_train, X2_test, y2_train, y2_test,
                                 '数据集2 (低分离度)', 'out')

    print('\n' + '=' * 60)
    print('实验总结')
    print('=' * 60)
    print(f'数据集1 (高分离度): 测试集错误率={error1:.4f}')
    print(f'数据集2 (低分离度): 测试集错误率={error2:.4f}')
    print('\n实验四完成！')

