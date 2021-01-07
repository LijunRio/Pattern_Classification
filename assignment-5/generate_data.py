import numpy as np
import matplotlib.pyplot as plt


def generate_sample():
    # 设置Sigma
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 设置mu
    mu_1 = np.array([1.0, -1.0])
    mu_2 = np.array([5.5, -4.5])
    mu_3 = np.array([1.0, 4.0])
    mu_4 = np.array([6.0, 4.5])
    mu_5 = np.array([9.0, 0.0])
    # 随机生成数据
    x_1 = np.random.multivariate_normal(mu_1, sigma, 200)
    x_2 = np.random.multivariate_normal(mu_2, sigma, 200)
    x_3 = np.random.multivariate_normal(mu_3, sigma, 200)
    x_4 = np.random.multivariate_normal(mu_4, sigma, 200)
    x_5 = np.random.multivariate_normal(mu_5, sigma, 200)

    x = np.concatenate([x_1, x_2], axis=0)
    x = np.concatenate([x, x_3], axis=0)
    x = np.concatenate([x, x_4], axis=0)
    x = np.concatenate([x, x_5], axis=0)

    # 数据可视化
    plt.scatter(x_1[:, 0], x_1[:, 1], marker='.', color='red')
    plt.scatter(x_2[:, 0], x_2[:, 1], marker='.', color='blue')
    plt.scatter(x_3[:, 0], x_3[:, 1], marker='.', color='black')
    plt.scatter(x_4[:, 0], x_4[:, 1], marker='.', color='green')
    plt.scatter(x_5[:, 0], x_5[:, 1], marker='.', color='purple')

    np.save('kmeans_data.npy', x)
    return x
