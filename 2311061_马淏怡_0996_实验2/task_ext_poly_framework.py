#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：拓展任务 多项式回归（单特征可视化）
- 目的：让学生实现多项式特征生成与线性回归闭式解，对比不同阶数
- 本模板保留数据处理与作图骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn
- 数据源：dataset_regression.csv（第一个数据集）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'dataset_regression.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task_ext_poly_fits_template.png')

np.random.seed(42)


def train_test_split_1d(x: np.ndarray, y: np.ndarray, test_ratio=0.2):
    n = x.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return x[idx[test_size:]], y[idx[test_size:]], x[idx[:test_size]], y[idx[:test_size]]


# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现 poly_features(x, degree) 与闭式解 fit_closed_form(X, y)
def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    # YOUR CODE HERE: 返回形如 [1, x, x^2, ..., x^degree]
    # 构造特征矩阵 X。结果形如 [x^0, x^1, x^2, ..., x^degree]
    # X_poly 形状为 (N, degree + 1)，其中 N 为样本数
    X_poly = np.ones((x.shape[0], degree + 1))

    # 从 i=1 开始填充 x^1, x^2, ..., x^degree
    for i in range(1, degree + 1):
        X_poly[:, i] = x ** i

    return X_poly


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # YOUR CODE HERE: 返回 theta = (X^T X)^{-1} X^T y；建议使用 pinv 增稳
    # 确保 y 是 N x 1 向量
    y_mat = y.reshape(-1, 1)

    # 使用 np.linalg.pinv (伪逆) 求解：theta = (X^T X)^-1 X^T y
    # 伪逆在矩阵不可逆或接近奇异时更稳定。
    XTX_pinv = np.linalg.pinv(X.T @ X)

    # 计算最终参数 theta
    theta = XTX_pinv @ X.T @ y_mat

    return theta.flatten()  # 返回一维向量
# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 读取两列数值：第一列为x，第二列为y
    df = pd.read_csv(CSV)
    num_df = df.select_dtypes(include=[np.number])
    x = num_df.iloc[:, 0].to_numpy().astype(float)
    y = num_df.iloc[:, 1].to_numpy().astype(float)

    # 划分与标准化（仅特征）
    x_train, y_train, x_test, y_test = train_test_split_1d(x, y, test_ratio=0.2)
    mean, std = x_train.mean(), x_train.std() + 1e-8
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    print('[Template] Implement poly_features and fit_closed_form then plot fits.')

    # 如已实现，可取消注释进行多阶数绘图
    degrees = [1, 2, 4]
    plt.figure(figsize=(7, 5))
    plt.scatter(x_train_std, y_train, s=8, color='#1f77b4', alpha=0.6, label='Train scatter')
    xs = np.linspace(np.min(x_train_std), np.max(x_train_std), 400)
    for deg in degrees:
        X_train_poly = poly_features(x_train_std, deg)
        theta = fit_closed_form(X_train_poly, y_train)
        train_m = float(np.mean((y_train - X_train_poly @ theta) ** 2))
        Xs_poly = poly_features(xs, deg)
        ys = Xs_poly @ theta
        plt.plot(xs, ys, label=f'degree={deg} (train {train_m:.3f})')
    plt.xlabel('standardized x')
    plt.ylabel('y')
    plt.title('Polynomial regression fits by degree')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()