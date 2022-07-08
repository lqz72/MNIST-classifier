import torch
import torchvision
import torch.nn.functional as F
from argparse import ArgumentParser
from network import FanNet
from metrics import *
import time
import yaml
import warnings

warnings.filterwarnings('ignore')
cfg = None

# 如果GPU可用 使用GPU加速训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个卷积神经网络
net = FanNet().to(device)


def load_data(data_path):
    """
    加载测试集数据
    :param data_path: 测试集路径
    :return: None
    """
    test_data = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=False,
    )

    return test_data


def evaluation(weight, test_data):
    """
    测试集评测模型
    :param test_data:测试数据
    :return: None
    """
    X_test = torch.unsqueeze(test_data.train_data, dim=1).float().to(device)
    y_test = test_data.test_labels.to(device)

    net.load_state_dict(torch.load(weight))
    net.eval()

    y_pred = torch.max(F.softmax(net(X_test), dim=1), 1)[1]

    # 计算评价指标
    labels = list(range(10))
    matrix = confusion_matrix(y_test, y_pred, labels)
    # print(matrix)
    P, R = precision(matrix), recall(matrix)
    F1 = F1_score(P, R)
    print("accuracy: {:.4f}".format((y_pred == y_test).sum() / len(y_test)))
    print("测试集评价指标：精度：{:.4f} 召回率：{:.4f} F1值：{:.4f}".format(P, R, F1))

    # img = torchvision.utils.make_grid(inputs)
    # img = img.numpy().transpose(1, 2, 0)

    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    # cv2.imshow('win', img)  # opencv显示需要识别的数据图片
    # key_pressed = cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', default='None', type=str)
    parser.add_argument('--data_path', default='./dataset/', type=str)
    parser.add_argument('--weight', default='./weight.pkl', type=str)
    args = parser.parse_args()

    if args.cfg != 'None':
        with open(args.cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        data_path = cfg['data_path']
        weight = cfg['weight']
    else:
        data_path = args.data_path
        weight = args.weight

    test_data = load_data(data_path)
    print('Predict begin...')
    start = time.time()
    evaluation(weight, test_data)
    end = time.time()
    print('预测时间', end - start)