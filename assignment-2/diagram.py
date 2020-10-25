import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.6, 1, 100)  # 从(-1,1)均匀取50个点
# y1 = pow((1- (1 / pow(5, r/10))), 5)
# y2 = pow((1- (1 / pow(10, r/10))), 10)
# y3 = pow((1- (1 / pow(100, r/10))), 100)

y1 = 1/pow(x, 5)


# y1 = (1/(pow(5, 1-r/10)))
# y2 = (1/(pow(10, 1-r/10)))
# y3 = (1/(pow(100, 1-r/10)))

plt.xlim((0, 1))

plt.plot(x, y1, label="n=5")
# plt.plot(r, y2, label="n=10")
# plt.plot(r, y3, label="n=100")
# plt.legend(loc=0,ncol=1)
plt.vlines(0.6, 0, 15, linestyles = "dashed")
plt.show()

