import numpy as np
import matplotlib.pyplot as plt
from network import net
from Dataset import get_dataset

if __name__ == "__main__":
    # 输入数据
    data_1 = [[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
              [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
              [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
              [-0.76, 0.84, -1.96]]
    data_2 = [[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
              [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
              [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
              [0.46, 1.49, 0.68]]
    data_3 = [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
              [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
              [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
              [0.66, -0.45, 0.08]]
    # 生成训练数据
    train_data, train_label = get_dataset([data_1, data_2, data_3], True)

    # 实验一: 隐含层不同结点数目对训练精度的影响
    # hids = [5, 10, 15, 20, 50, 100, 150]
    # hids_str = [str(x) for x in hids]
    # print(hids_str)
    # loss_l = []
    # bar_width = 0.4
    # for hid in hids:
    #     model = net(train_data, train_label, h_num=hid)
    #     e = model.train(bk_mode='single', eta=1e-1, epoch_num=1000)
    #     plt.plot(e, label="{}".format(hid))  # 效果一样的
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=10, mode="expand", borderaxespad=0.)
    # plt.savefig('experiment1.png')
    # plt.show()

    # 实验二: 观察不同的梯度更新步长对训练的影响，并给出一些描述或解释；
    # etas = [1e-3, 1e-2, 1e-1, 5e-1, 1]
    # etas_str = [str(x) for x in etas]
    # print(etas_str)
    # loss_l = []
    # bar_width = 0.4
    # for eta in etas:
    #     model = net(train_data, train_label, h_num=10)
    #     e = model.train(bk_mode='single', eta=eta, epoch_num=1000)
    #     plt.plot(e, label="{}".format(eta))  # 效果一样的
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=10, mode="expand", borderaxespad=0.)
    # plt.savefig('experiment2.png')
    # plt.show()

    # 实验三: 在网络结构固定的情况下，绘制出目标函数随着迭代步数增加的变化曲线
    model1 = net(train_data, train_label, h_num=10)
    e1 = model1.train(bk_mode='single', eta=0.6, epoch_num=300)
    plt.plot(e1, label="{}".format('single'))  # 效果一样的
    model2 = net(train_data, train_label, h_num=10)
    e2 = model2.train(bk_mode='batch', eta=0.6, epoch_num=300)
    plt.plot(e2, label="{}".format('batch'))  # 效果一样的

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=10, mode="expand", borderaxespad=0.)
    plt.savefig('experiment3.png')
    plt.show()
