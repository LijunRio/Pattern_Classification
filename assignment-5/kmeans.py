import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_sample


def k_means(data, mu):
    """
    K-means聚类
    :param data: 待聚类数据(np.array)
    :param mu:  初始化聚类中心(np.array)
    :return:
        class_result: 聚类结果[[第一类数据], [第二类数据], ... , [第c类数据]]
        label: 分类结果
        mu: 类中心结果[第一类类中心, 第二类类中心, ... , 第c类类中心]
        iter_num: 迭代次数
    """
    # 待聚类数据矩阵调整(复制矩阵使其从n*d变为n*c*d, c为中心点mu的数目)
    # (1000, 2)->(1000, 5, 2)
    data = np.tile(np.expand_dims(data, axis=1), (1, mu.shape[0], 1))
    mu_temp = np.zeros_like(mu)  # 保存前一次mu的结果
    iter_num = 0

    while np.sum(mu - mu_temp):
        mu_temp = mu
        iter_num += 1
        label = np.zeros((data.shape[0]), dtype=np.uint8)

        # 调整矩阵mu与data的格式一致 (5, 2)->(1000, 5, 2)
        mu = np.tile(np.expand_dims(mu, axis=0), (data.shape[0], 1, 1))
        # 生成距离矩阵(1000, 5)
        dist = np.sum(pow((data - mu), 2), axis=-1)

        class_result = []  # 是五个类别
        for i in range(data.shape[1]):
            class_result.append([])

        for index, sample in enumerate(data):
            minDist_index = np.argmin(dist[index])
            # sample为五个重复的数据，所以取第一个就行了
            class_result[minDist_index].append(sample[0])
            label[index] = minDist_index
        class_result = np.array(class_result)
        mu = []
        for i in class_result:
            new_mean = np.mean(i, axis=0)
            mu.append(new_mean)
        mu = np.array(mu)

    return class_result, label, mu, iter_num


if __name__ == '__main__':
    data = np.load('kmeans_data.npy')  # (1000,  2)
    mu_gt = np.array([[1.0, -1.0], [5.5, -4.5], [1.0, 4.0], [6.0, 4.5], [9.0, 0.0]])
    label_gt = []
    for i in range(mu_gt.shape[0]):
        for j in range(200):
            label_gt.append(j)
    label_gt = np.array(label_gt)

    # 随机初始化5组中心进行聚类结果分析
    for r in range(4):
        mu_input = mu_gt + np.round(((r + 1)) * np.random.rand(5, 2), 2)
        class_result, label, result_mu, iter = k_means(data, mu_input)
        print("第",r+1,"组实验-------------")
        print(" 共迭代了{}次".format(iter))
        # 1. 统计错分样本
        mis_class = 0
        for k in range(label.shape[0]):
            if label[k] == label_gt[k]:
                mis_class += 1
        print(" 错误分类样本数为:", mis_class)

        E = 0
        color = ['red', 'blue', 'black', 'green', 'purple']
        for idx, i in enumerate(class_result):
            i = np.array(i)
            e = np.matmul((result_mu[idx] - mu_gt[idx]).T, (result_mu[idx] - mu_gt[idx]))
            E += e
            print(" 第{}类: 初始化类中心{}, 结果为{}, 样本数为{}, 聚类中心均方误差为{}".format(idx, mu_input[idx], result_mu[idx], i.shape[0], e))
            plt.scatter(i[:, 0], i[:, 1], marker='.', color=color[idx])
            plt.scatter(result_mu[idx, 0], result_mu[idx, 1], s=200, marker='o', c='c', cmap='coolwarm')
            plt.scatter(mu_gt[idx, 0], mu_gt[idx, 1], s=200, marker='x', c='r', cmap='coolwarm')
        print(" 聚类整体均方误差和为{}".format(E))
        name = 'No.'+str((r+1))+' experiment'
        plt.title(name)
        plt.savefig(name+'.png')
        plt.show()
