from sklearn.model_selection import learning_curve
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score
# 机器学习模型
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from metrics import *

import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01), cv=10,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('./{}.jpg'.format(title))


def load_data():
    """导入手写数字数据
    """

    train_dataset = datasets.MNIST(
        root='./dataset',
        train=True,
        transform=transforms.ToTensor(),
        # download=True
    )
    test_dataset = datasets.MNIST(
        root='./dataset',
        train=False,
        transform=transforms.ToTensor(),
        # download=True
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    trian_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    X_train = trian_images.numpy().reshape(-1, 28*28)
    y_train = train_labels.numpy().reshape(-1,)

    X_test = test_images.numpy().reshape(-1, 28*28)
    y_test = test_labels.numpy().reshape(-1, )

    return X_train, X_test, y_train, y_test


def evalution(clf, X_test, y_test):
    """测试集评测
    """
    y_pred = clf.predict(X_test)
    labels = [i for i in range(10)]
    matrix = confusion_matrix(y_test, y_pred, labels)
    P, R = precision(matrix), recall(matrix)
    F1 = F1_score(P, R)

    print('测试集评价指标：')
    test_acc = accuracy_score(y_test, y_test)
    print('accuracy: ', test_acc)
    print("精度：{:.4f} 召回率：{:.4f} F1值：{:.4f}".format(P, R, F1))


def libsvm_exp(X_train, X_test, y_train, y_test):
    """SVM实验
    """
    start = time.time()

    X = np.concatenate((X_train, X_test))
    feature_map_nystroem = Nystroem(n_components=28 * 28)
    feature_map_nystroem.fit(X)
    X = feature_map_nystroem.transform(X)

    X_train, X_test = X[:60000], X[60000:]

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    end = time.time()

    print('耗时:', end - start, 's')
    train_acc = accuracy_score(y_train, y_pred)
    print('accuracy: ', train_acc)

    evalution(clf, X_test, y_test)


def svm_exp(X_train, X_test, y_train, y_test):
    """SVM实验
    """
    start = time.time()
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    end = time.time()

    print('耗时:', end - start, 's')
    train_acc = accuracy_score(y_train, y_pred)
    print('accuracy: ', train_acc)

    evalution(clf, X_test, y_test)


def dt_exp(X_train, X_test, y_train, y_test):
    """决策树实验
    """
    start = time.time()
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    end = time.time()

    print('耗时:', end - start, 's')
    train_acc = accuracy_score(y_train, y_pred)
    print('accuracy: ', train_acc)

    labels = [i for i in range(10)]
    matrix = confusion_matrix(y_train, y_pred, labels)
    P, R = precision(matrix), recall(matrix)
    F1 = F1_score(P, R)

    print('训练集评价指标：')
    print("精度：{:.4f} 召回率：{:.4f} F1值：{:.4f}".format(P, R, F1))

    evalution(clf, X_test, y_test)


def random_forest_exp(X_train, X_test, y_train, y_test):
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    end = time.time()

    print('耗时:', end - start, 's')
    train_acc = accuracy_score(y_train, y_pred)
    print('accuracy: ', train_acc)

    evalution(clf, X_test, y_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    dt_exp(X_train, X_test, y_train, y_test)

    # a = np.array([
    #     [0.9891, 0.9893, 0.9892],
    #     [0.9792, 0.9791, 0.9791],
    #     [0.9692, 0.9691, 0.9692],
    #     [0.9353, 0.9353, 0.9353],
    #     [0.8769, 0.8767, 0.8768],
    # ])
    #
    # b = [301.8, 775.3,  79.7, 42.9, 22.0]
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    #
    # plt.figure()
    # plt.title('不同模型的执行耗时')
    # plt.xlabel('model')
    # plt.ylabel('time/s')
    # labels = ['CNN', 'SVC', 'Random Forest', 'LinearSVC', 'Decision Tree', ]
    # # labels = ['SVC', 'CNN', 'Random Forest', 'LinearSVC', 'Decision Tree', ]
    # metircs = ['precision', 'recall', 'F1-score']
    #
    # plt.xticks(range(5), labels)
    # plt.plot(range(5), b, marker='*', color='g')
    # # plt.legend()
    # plt.savefig('./model_time.jpg')
    # plt.show()