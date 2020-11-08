import data as data
import numpy as np

data_size = 10

sample1 = np.array(data.sample2)
sample2 = np.array(data.sample4)
norm = np.ones([data_size, 1])

# 按列连接两个矩阵，变为规范化增广形式
sample1 = np.c_[sample1, norm]
sample2 = -np.c_[sample2, norm]

# 按行连接两个矩阵，使样本集变为一个
Y = np.r_[sample1, sample2]

# 初始化
learning_rate = 1
a = np.zeros(Y.shape[1])  # [1*3]
b = np.random.rand(Y.shape[0])  # [1*20]
e = np.dot(a, Y.transpose()) - b  # Ya-b
# print(Y.shape)
# print(np.linalg.pinv(Y).shape)

iter_num = 0
while min(e) < 0:
    iter_num += 1
    e_ = 0.5 * (e + abs(e))
    b = b + 2 * learning_rate * e_
    a = np.dot(b, np.linalg.pinv(Y).transpose())
    e = np.dot(a, Y.transpose()) - b
    print("iteration nums:" + str(iter_num) + ", a = " + str(a) + ", b = " + str(b)+"  ")
    if iter_num > 50000:
        break
    if abs(e).sum() < 0.001:
        break
print(e)
