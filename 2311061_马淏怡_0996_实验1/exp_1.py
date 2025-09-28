
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# 数据加载
def Img2Mat0(filename):

    raw = np.loadtxt(filename)
    X = raw[:, :256].astype(float)
    Y_onehot = raw[:, 256:]
    y = np.argmax(Y_onehot, axis=1).astype(int)
    return X, y


#  手写版 KNN
def classify0(inX, dataSet, labels, k):

    diff = dataSet - inX
    dist = np.sqrt(np.sum(diff ** 2, axis=1))  # 欧式距离
    idx = np.argsort(dist)  # 排序
    votes = {}
    for i in range(k):
        lab = labels[idx[i]]
        votes[lab] = votes.get(lab, 0) + 1
    # 返回票数最多的类别（若平局，取最小类别编号）
    max_votes = max(votes.values())
    winners = [lab for lab, v in votes.items() if v == max_votes]
    return int(min(winners))


def loo_eval_handmade(X, y, k):
    """手写版 LOO 交叉验证"""
    n = X.shape[0]
    correct = 0
    for i in range(n):
        X_train = np.vstack((X[:i], X[i + 1:]))
        y_train = np.hstack((y[:i], y[i + 1:]))
        pred = classify0(X[i], X_train, y_train, k)
        if pred == y[i]:
            correct += 1
    return correct / n


# ========== 主函数 ==========
if __name__ == "__main__":
    X, y = Img2Mat0("semeion.data")

    print("\n===== 自写版 KNN LOO 结果 =====")
    for k in [1, 3, 5]:
        acc = loo_eval_handmade(X, y, k)
        print(f"k={k}, 准确率={acc:.4f}")
