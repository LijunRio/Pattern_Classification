import numpy as np
import data as data

# 测试集制作
sample1 = np.array(data.sample1)[:8, :]
sample2 = np.array(data.sample2)[:8, :]
sample3 = np.array(data.sample3)[:8, :]
sample4 = np.array(data.sample4)[:8, :]
norm = np.ones(sample1.shape[0])

sample1 = np.c_[sample1, norm]
sample2 = np.c_[sample2, norm]
sample3 = np.c_[sample3, norm]
sample4 = np.c_[sample4, norm]

x = np.r_[sample1, sample2, sample3, sample4]
x = x.transpose()


one = np.ones(sample1.shape[0])
zero = np.zeros(sample1.shape[0])
y = np.r_[np.c_[one, zero, zero, zero], np.c_[zero, one, zero, zero], np.c_[zero, zero, one, zero], np.c_[
    zero, zero, zero, one]]
y = y.transpose()

w = np.dot(np.dot(y, x.transpose()), np.linalg.inv(np.dot(x, x.transpose())))

print("$\hat w =$ " + str(w))

test1 = np.array(data.sample1)[8:, :]
test2 = np.array(data.sample2)[8:, :]
test3 = np.array(data.sample3)[8:, :]
test4 = np.array(data.sample4)[8:, :]
norm = np.ones(test1.shape[0])

test1 = np.c_[test1, norm]
test2 = np.c_[test2, norm]
test3 = np.c_[test3, norm]
test4 = np.c_[test4, norm]
_x = np.r_[test1, test2, test3, test4]
_y = np.dot(_x, w.transpose())
_y = np.argmax(_y, axis=1) + 1
print("test_result:", _y)

