# import numpy as np
# import data as data
#
# data_size = 10
#
# sample1 = np.array(data.sample1)
# sample2 = np.array(data.sample2)
# # sample1 = np.array(data.sample3)
# # sample2 = np.array(data.sample4)
# norm = np.ones([data_size, 1])
#
# # 按列连接两个矩阵，变为规范化增广形式
# sample1 = np.c_[sample1, norm]
# sample2 = -np.c_[sample2, norm]
#
# # 按行连接两个矩阵，使样本集变为一个
# sample = np.r_[sample1, sample2]
# # 对样本矩阵求转置
# sample_transpose = sample.transpose()
# # 构造a矩阵
# a = np.zeros(sample_transpose.shape[0])
#
# # 计算J_p(a)
# y = np.dot(a, sample_transpose)
#
# iter_num = 0
# while min(y) <= 0:
#     iter_num += 1
#     for i in range(y.shape[0]):
#         if y[i] <= 0:
#             a = a + sample[i]
#     y = np.dot(a, sample_transpose)
#     print("Iteration nums:" + str(iter_num) + ", $a^T$ = " + str(a)+"  ")

import numpy as np
import data as data

data_size = 4
sample= np.array([[1, 4], [2, 3], [-4, -1], [-3,-2]])

norm = np.array([1, 1, -1, -1])
# 按列连接两个矩阵，变为规范化增广形式
sample = np.c_[sample, norm]
# sample = np.c_[norm, sample]

# 对样本矩阵求转置
sample_transpose = sample.transpose()
# 构造a矩阵
a = [0, 1, 0]
print(sample_transpose)

# 计算J_p(a)
y = np.dot(a, sample_transpose)

iter_num = 0
while min(y) <= 0:
    iter_num += 1
    for i in range(y.shape[0]):
        if y[i] <= 0:
            a = a + sample[i]
    y = np.dot(a, sample_transpose)
    print("Iteration nums:" + str(iter_num) + ", $a^T$ = " + str(a)+"  ")
