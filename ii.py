import numpy as np
import matplotlib.pyplot as plt
import math
import pca


def opentxt(filename):
    a = list()
    fp = open(filename).readlines()
    for i in fp:
        a.append([float(i.split('\t')[0]), float(i.split('\t')[1])])
    return a


def M(a):  # 平均值函数
    m = np.mean(a, axis=0)
    return m


def S(a):  # 离散度矩阵
    m = M(a)
    sum = np.matrix([[0, 0], [0, 0]])
    # a=np.matrix(a)
    for i in a:
        print(np.dot(np.matrix((i - m)).T, np.matrix((i - m))))
        sum = np.dot(np.matrix((i - m)).T, np.matrix((i - m))) + sum
    return sum


def main():
    a = pca.opentxt1('boy.txt')
    b = pca.opentxt1('girl.txt')
    for i in b:
        a.append(i)
    a=np.array(a)
    b=np.cov(a.T)
    A, B = np.linalg.eig(b)#A为特征值，B为特征向量
    x=pca.PCA(A,B)#投影方向
    a=np.dot(a,x)
    ori = list()
    m1 = M(a)
    m2 = M(b)
    s1 = S(a)
    s2 = S(b)
    sw = s1 + s2
    sw = np.matrix(sw).I
    m3 = np.matrix((m1 - m2)).T
    w = np.dot(sw, m3)*1000 # t投影方向 二维列向
    # print(w)
    c = opentxt('boy82.txt')
    d = opentxt('girl42.txt')
    w0 = (m1 + m2) / 2 * (-1)
    x = [i for i in np.arange(160, 175, 0.1)]
    y = list()
    w1 = np.dot(w0, w)

    plt.plot([100*float(w[0]),130*float(w[0])],[100*float(w[1]),130*float(w[1])])
    for i in x:
        y.append(-(float(w[0]) * i + float(w1)) / float(w[1]))
    plt.plot(x, y)
    l = 0
    ans = list()
    for i in c:
        ori.append(1)
    x1 = [row[0] for row in c]
    y1 = [row[1] for row in c]
    plt.plot(x1, y1, 'o', label='$line$', linewidth=1)
    for i in d:
        c.append(i)
        ori.append(0)
    x = [row[0] for row in d]
    y = [row[1] for row in d]
    plt.plot(x, y, 'o', label='$line$', linewidth=1)
    plt.show()
    for i in c:
        if (np.dot((i + w0), w) > l):
            ans.append(1)
        else:
            ans.append(0)


main()
