"""
Author: Yongfu Fan
Time: 2022/6/6 13:27
版权所有 违反必究
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch
import torchvision
import time
import yaml

from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from pytorchtools import EarlyStopping
from metrics import *
from network import FanNet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同
cfg = None


def load_data(args, valid=True):
    """
    加载数据集和配置文件
    :param args: 训练参数
    :param valid: 是否划分验证集
    :return:
    """
    if args.cfg != 'None':
        with open(args.cfg, 'r') as f:
            global cfg
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        batch_size = cfg['batch_size']
        data_path = cfg['data_path']
    else:
        batch_size = args.batch_size
        data_path = args.data_path

    # 训练集
    train_data = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    if not valid:
        #  如果不需要验证集 直接返回
        return Data.DataLoader(train_data, batch_size, shuffle=True), None

    # 将训练集进一步划分为训练集和验证集
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = Data.DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = Data.DataLoader(valid_data, len(valid_data), shuffle=False)

    return train_loader, valid_loader


def get_optimizer(cnn, optimize='SGD', lr=0.01):
    """
    获取对应的优化器

    :param cnn: 网络模型
    :param optimize: 优化器
    :param lr: 学习率
    :return: None
    """
    if optimize == 'SGD':
        return torch.optim.SGD(cnn.parameters(), lr=lr, weight_decay=1e-4)
    elif optimize == 'Momentum':
        return torch.optim.SGD(cnn.parameters(), momentum=0.5, lr=lr, weight_decay=1e-4)
    elif optimize == 'NAG':
        return torch.optim.SGD(cnn.parameters(), momentum=0.5, nesterov=True, lr=lr, weight_decay=1e-4)
    elif optimize == 'Adam':
        return torch.optim.Adam(cnn.parameters(), eps=1e-8, lr=lr, weight_decay=1e-4)
    elif optimize == 'AdaGrad':
        return torch.optim.Adagrad(cnn.parameters(), eps=1e-8, lr=lr, weight_decay=1e-4)
    elif optimize == 'RMSProp':
        return torch.optim.RMSprop(cnn.parameters(), eps=1e-8, lr=lr, weight_decay=1e-4)
    elif optimize == 'AdaDelta':
        return torch.optim.Adadelta(cnn.parameters(), eps=1e-8, lr=lr, weight_decay=1e-4)
    elif optimize == 'NAdam':
        return torch.optim.NAdam(cnn.parameters(), eps=1e-8, lr=lr)


# 如果GPU可用 使用GPU加速训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个卷积神经网络
net = FanNet().to(device)


def train(args, train_loader, valid_loader):
    """
    进行卷积神经网络的训练
    :param args: 训练超参数
    :param train_loader: 训练数据
    :param valid_loader: 验证数据
    :return:
    """

    if cfg is None:
        # 训练轮数
        epochs = args.epochs
        # 优化器和学习率
        learning_rate = args.learning_rate
        optimizer = get_optimizer(net, args.optimizer, lr=learning_rate)
    else:
        epochs = cfg['epochs']
        learning_rate = cfg['learning_rate']
        optimizer = get_optimizer(net, cfg['optimizer'], lr=learning_rate)

    # 损失函数
    loss_func = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping()
    writer = SummaryWriter('run')
    epoch_loss, epoch_metric = [], []

    iter = 0
    for epoch in range(1, epochs + 1):
        net.train()
        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            loss = loss_func(y_pred, y)  # 计算损失

            # 反向传播并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                y_pred = torch.max(F.softmax(y_pred, dim=1), 1)[1]
                accuracy = float((y_pred == y.data).sum()) / float(y.size(0))
                # print("epoch:{:2d} step:{:3d} | train loss：{:.2f} "
                #       "train accuracy: {:.2f}".format(epoch, step, loss, accuracy))
                writer.add_scalar('Train/Loss', loss, iter)
                writer.add_scalar('Train/Acc', accuracy, iter)
                iter += 1

        if not valid_loader:
            continue
        # 验证集只测试模型效果 不进行梯度更新
        with torch.no_grad():
            net.eval()
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                test_output = net(X)
                valid_loss = loss_func(test_output, y)

                y_pred = torch.max(F.softmax(test_output, dim=1), 1)[1]
                accuracy = float((y_pred == y).sum()) / float(y.size(0))

                labels = list(range(10))
                matrix = confusion_matrix(y.data, y_pred, labels)
                P, R = precision(matrix), recall(matrix)
                F1 = F1_score(P, R)

                print("epoch:{:2d} finish | accuracy: {:.2f}".format(epoch, accuracy))
                print("验证集评价指标：精度：{:.2f} 召回率：{:.2f} F1值：{:.2f}".format(P, R, F1))

                epoch_loss.append(valid_loss)
                epoch_metric.append([P, R, F1])
                early_stopping(valid_loss, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    writer.close()

    # 保存训练的模型
    torch.save(net.state_dict(), 'weight.pkl')

    # plot(epoch_loss, epoch_metric, epochs, learning_rate, args.optimizer)
    # metric2csv(epoch_metric)


def metric2csv(epoch_metric):
    """
    将每一轮的评价指标转换为csv格式
    :param epoch_metric: 每个epoch的评测指标
    :return: None
    """
    P = np.array(epoch_metric)[:, 0].reshape(-1, )
    df = pd.DataFrame(data={'{}'.format(args.optimizer): P})
    df.to_csv('./{}-precision.csv'.format(args.optimizer), index=0)

    R = np.array(epoch_metric)[:, 1].reshape(-1, )
    df = pd.DataFrame(data={'{}'.format(args.optimizer): R})
    df.to_csv('./{}-recall.csv'.format(args.optimizer), index=0)

    F1 = np.array(epoch_metric)[:, 2].reshape(-1,)
    df = pd.DataFrame(data={'{}'.format(args.optimizer): F1})
    df.to_csv('./{}-F1.csv'.format(args.optimizer), index=0)


def plot(epoch_loss, epoch_metric, epochs, learning_rate, optimizer):
    """
    绘制训练结果
    :param epoch_loss:损失数组
    :param epoch_metric:评价指标数组
    :param epochs:训练轮数
    :param learning_rate:学习率 str
    :param optimizer: 优化器名称 str
    :return: None
    """
    plt.figure()
    plt.title('{} Loss Curve'.format(optimizer))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.savefig('./result/valid_loss_{}_{}.jpg'.format(epochs, optimizer))

    epoch_metric = np.array(epoch_metric)
    plt.figure()
    plt.title('{} Metric Curve'.format(optimizer))
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.plot(range(len(epoch_metric)), epoch_metric[:, 0], label='precision')
    plt.plot(range(len(epoch_metric)), epoch_metric[:, 1], label='recall')
    plt.plot(range(len(epoch_metric)), epoch_metric[:, 2], label='F1-score')
    plt.legend()
    plt.savefig('./result/valid_metric_{}_{}.jpg'.format(epochs, optimizer))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', default='None', type=str)
    parser.add_argument('--data_path', default='./dataset/', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--activation', default='LeakyReLU', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--Regularization', default='None', type=str)
    args = parser.parse_args()

    # 准备数据集
    train_loader, valid_loader = load_data(args, valid=True)
    print('Train begin...')
    start = time.time()
    train(args, train_loader, valid_loader)
    end = time.time()
    print('训练时间：', end - start)

    # df = pd.read_csv('./csv/optimizer-R.csv', encoding='utf-8')
    # print(df)
    # data = np.array(df)
    # optimizers = df.columns
    #
    # plt.figure()
    # plt.title('基于不同优化器的模型验证集召回率')
    # plt.xlabel('epoch')
    # plt.ylabel('metric value')
    #
    # for i in range(data.shape[1]):
    #     plt.plot(data[:, i], label=optimizers[i])
    #
    # plt.legend()
    # plt.savefig('./optimizer-R.jpg')
    # plt.show()

    # a = [
    #     [0.99, 0.99, 0.99],
    #     [0.98, 0.98, 0.98],
    #     [0.94, 0.92, 0.93],
    #     [0.89, 0.87, 0.88],
    #     [0.87, 0.80, 0.83]
    # ]
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # a = np.array(a)
    # plt.figure()
    # plt.title('不同学习率的模型评价指标')
    # plt.xlabel('learning_rate')
    # plt.ylabel('metric value')
    # plt.xticks(range(5), [0.001, 0.002, 0.004, 0.008, 0.1])
    # plt.plot(a[:, 0], label='precision', marker='*')
    # plt.plot(a[:, 1], label='recall', marker='d')
    # plt.plot(a[:, 2], label='F1-score', marker='^')
    # plt.legend()
    # plt.savefig('./lr.jpg')
    # plt.show()

