import numpy as np
import pandas as pd


# 混淆矩阵
def confusion_matrix(y_true, y_pred, labels=[0, 1, 2]):
    # 默认转换为numpy数组
    # y_true, y_pred = np.array(y_true), np.array(y_pred)

    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                index = (y_true == labels[i])
                matrix[i][j] = sum(y_true[index] == y_pred[index])
            else:
                # wrongly predict i to j
                index = (y_true == labels[i])
                matrix[i][j] = sum(y_pred[index] == labels[j])

    return matrix


# 精度
def precision(conf_matrix):
    n = len(conf_matrix)
    P = np.zeros((n,))

    for i in range(n):
        P[i] = conf_matrix[i][i] / sum(conf_matrix[:, i])
    #     print('精度: ', P)
    return sum(P) / n


# 召回率
def recall(conf_matrix):
    n = len(conf_matrix)
    R = np.zeros((n,))

    for i in range(n):
        R[i] = conf_matrix[i][i] / sum(conf_matrix[i, :])
    #     print('召回率: ', R)
    return sum(R) / n


# F1度量
def F1_score(P, R):
    return 2 * P * R / (P + R)
