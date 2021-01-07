import numpy as np
import struct
# 使用python的struct模块来完成.可以用 struct来处理c语言中的结构体.
from sklearn import svm
from matplotlib import pyplot as plt
import pickle

"""
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255.
    0 means background (white), 255 means foreground (black).
"""


def load_imgs(path):
    """
    加载图像数据
    :param path: 图像数据文件路径
    :return: imgs: 返回图像数据(num, img.size) 每行为一个数据
    """
    with open(path, 'rb') as f:  # 'rb'以二进制读取文件
        #  读取image文件前4个整型数字
        #  unpack(fmt, string)按照给定的格式(fmt)解析字节流string，返回解析出来的tuple
        # >4I 大端模式 4个Int I: int 4*4=16
        # f.read(16)读取前16个字节的数据
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))  # 读完后缓冲区少了这16字节内容
        #  整个images数据大小为60000*28*28
        # imgs_size = num * rows *cols
        imgs = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)  # nums * img_size

    return imgs


"""
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    """


def load_labels(path):
    """
    加载label数据
    :param path: Lable数据文件路径
    :return: labels 每行为一个label的标签(n,) n*1维np.array对象，n为图片数量
    """
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>2I', f.read(8))
        #  读取label文件前2个整形数字，label的长度为num
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def generate_data(img_path, label_path, class_list):
    """
    生成指定类别的数据集
    :param img_path: 图像数据文件路径
    :param label_path: 类别标签数据路径
    :param class_list:  需要生成的类别号列表
    :return: [img_class, label_class]：
        img_class为图像数据数组,每行为一个图像的数据(n,img.size);
        label_class为标签数据数组,每行为一个标签的数据(n,)
    """

    # 加载图像数据和label
    imgs = load_imgs(img_path)
    labels = load_labels(label_path)

    # 选取特定类别生成所需要的数据
    img_class = []
    label_class = []
    for c in class_list:
        index = np.where(labels == c)  # 返回的是label为指定值的index，格式([index的list],dtype)
        img_class.extend(imgs[index[0]])
        label_class.extend((labels[index[0]]))
    # 将数据进行归一化，即图像幅值变为0-1
    img_class = np.array(img_class) / 255
    label_class = np.array(label_class)

    return [img_class, label_class]


if __name__ == "__main__":

    # 文件路径
    TRAIN_IMG_PATH = "./MNIST_data/train-images.idx3-ubyte"
    TRAIN_LABEL_PATH = "./MNIST_data/train-labels.idx1-ubyte"
    TEST_IMG_PATH = "./MNIST_data/t10k-images.idx3-ubyte"
    TEST_LABEL_PATH = "./MNIST_data/t10k-labels.idx1-ubyte"
    # 训练&测试的指定类别
    img_class = [1, 2]
    # 数据加载
    train_data = generate_data(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, img_class)
    test_data = generate_data(TEST_IMG_PATH, TEST_LABEL_PATH, img_class)

    # 1: 不同的超参数C对结果影响
    print('--------------1: 不同的C对结果影响-----------------')
    C = [0.0001, 0.001, 0.1, 1, 5, 10, 100, 1000, 10000]
    kernel = 'rbf'
    for i in C:
        s = svm.SVC(C=i, kernel=kernel)
        s.fit(train_data[0], train_data[1])
        y_pred = s.predict(test_data[0])
        print('C={}, kernel={}, accuracy={}'.format(i, kernel, s.score(test_data[0], test_data[1])))

    # 2: 不同的kernel对结果影响
    print('--------------2: 不同的kernel对结果影响 -----------------')
    C = 0.5
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in kernel:
        s = svm.SVC(C=C, kernel=i)
        s.fit(train_data[0], train_data[1])
        y_pred = s.predict(test_data[0])
        print('C={}, kernel={}, accuracy={}'.format(C, i, s.score(test_data[0], test_data[1])))

