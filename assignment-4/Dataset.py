import numpy as np
import matplotlib.pyplot as plt
import random


def get_dataset(data_input, shuffle):
    """
    根据输入数据, 生成相应的label, 形成训练数据
    Parameter:
        data_input: 输入数据列表 [[类1数据], [类2数据], [类3数据], ...]
    Return:
        train_data: 训练用数据列表 [[数据1], [数据2], ...]
        train_label: 训练用Label列表 [[数据1对应Label], [数据2对应Label], ...]
    """
    train_data = []
    train_label = []
    for index, item in enumerate(data_input):
        for j in item:
            # 数据列表
            data = np.array(j)
            train_data.append(data)
            # Label列表: 对应类别为1, 其余为0
            label = np.zeros_like(data)
            label[index] = 1
            train_label.append(label)
    if shuffle:
        sample = list(zip(train_data, train_label))
        random.shuffle(sample)
        train_data[:], train_label[:] = zip(*sample)
    return train_data, train_label


