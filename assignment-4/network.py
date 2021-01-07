import numpy as np
import matplotlib.pyplot as plt


class net:
    """
    三层网络类
    """

    def __init__(self, train_data, train_label, h_num):
        """
        网络初始化
        Parameters:
            train_data: 训练用数据列表
            train_label: 训练用Label列表
            h_num: 隐含层结点数
        """
        # 初始化数据
        self.train_data = train_data  # 30个样本的list
        self.train_label = train_label
        self.h_num = h_num
        # 随机初始化权重矩阵
        self.w_ih = np.random.rand(train_data[0].shape[0], h_num)  # 3, hnum
        self.w_hj = np.random.rand(h_num, train_label[0].shape[0])  # hnum, 3
        # print(self.w_ih.shape)
        # print(self.w_hj.shape)

    def tanh(self, data):
        """
        tanh函数
        """
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def sigmoid(self, data):
        """
        Sigmoid函数
        """
        return 1 / (1 + np.exp(-data))

    def forward(self, data):
        """
        前向传播
        Parameter:
            data: 单个样本输入数据
        Return:
            z_j: 单个输入数据对应的网络输出
            y_h: 对应的隐含层输出, 用于后续反向传播时权重更新矩阵的计算
        """
        # 计算隐含层输出
        net_h = np.matmul(data.T, self.w_ih)  # 两个numpy数组的矩阵相乘
        y_h = self.tanh(net_h)
        # 计算输出层输出
        net_j = np.matmul(y_h.T, self.w_hj)
        z_j = self.sigmoid(net_j)

        return z_j, y_h

    def backward(self, z, label, eta, y_h, x_i):
        """
        反向传播
        Parameters:
            z: 前向传播计算的网络输出
            label: 对应的Label
            eta: 学习率
            y_h: 对应的隐含层输出
            x_i: 对应的输入数据
        Return:
            delta_w_hj: 隐含层-输出层权重更新矩阵
            delta_w_ih: 输入层-隐含层权重更新矩阵
            error: 样本输出误差, 用于后续可视化
        """
        # 矩阵维度整理
        z = np.reshape(z, (z.shape[0], 1))
        label = np.reshape(label, (label.shape[0], 1))
        y_h = np.reshape(y_h, (y_h.shape[0], 1))
        x_i = np.reshape(x_i, (x_i.shape[0], 1))
        # 计算输出误差
        error = np.matmul((label - z).T, (label - z))[0][0]
        # 计算隐含层-输出层权重更新矩阵
        error_j = (label - z) * z * (1 - z)
        delta_w_hj = eta * np.matmul(y_h, error_j.T)
        # 计算输入层-隐含层权重更新矩阵
        error_h = np.matmul(((label - z) * z * (1 - z)).T, self.w_hj.T).T * (1 - y_h ** 2)
        delta_w_ih = eta * np.matmul(x_i, error_h.T)

        return delta_w_hj, delta_w_ih, error

    def train(self, bk_mode, eta, epoch_num):
        """
        网络训练
        Parameters:
            bk_mode: 反向传播方式 single or batch
            eta: 学习率
            epoch_num: 全部训练数据迭代次数
        """
        # 单样本更新
        if bk_mode == 'single':
            E = []
            for epoch in range(epoch_num):
                e = []
                for idx, x_i in enumerate(self.train_data):
                    # 前向传播
                    z, y_h = self.forward(x_i)
                    # 反向传播
                    delta_w_hj, delta_w_ih, error = self.backward(z, self.train_label[idx], eta, y_h, x_i)
                    # 权重矩阵更新
                    self.w_hj += delta_w_hj
                    self.w_ih += delta_w_ih
                    print("sample:", idx, x_i, "error: ", round(error, 2))
                    e.append(error)
                print("iteration nums:", epoch, " mean error:", np.mean(e), "=====================================")
                E.append(np.mean(e))  # 每30个所有数据的平均误差再append进去

        # 批次更新
        if bk_mode == 'batch':
            E = []
            for epoch in range(epoch_num):
                e = []
                Delta_w_hj = 0
                Delta_w_ih = 0
                for idx, x_i in enumerate(self.train_data):
                    # 前向传播
                    z, y_h = self.forward(x_i)
                    # 反向传播
                    delta_w_hj, delta_w_ih, error = self.backward(z, self.train_label[idx], eta, y_h, x_i)
                    # 更新权重矩阵累加
                    Delta_w_hj += delta_w_hj
                    Delta_w_ih += delta_w_ih
                    e.append(error)
                # 权重矩阵批次更新
                self.w_hj += Delta_w_hj
                self.w_ih += Delta_w_ih
                print("iteration nums:", epoch, " mean error:", np.mean(e), "=====================================")
                E.append(np.mean(e))

        # 可视化迭代优化过程
        # plt.title('iteration progress')
        # plt.plot(E)
        return E
        # plt.plot(E, label="{}".format(self.h_num))  # 效果一样的
        # plt.show()
