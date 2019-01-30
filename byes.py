import matplotlib as mpl
import tkinter
import numpy
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
def ROC(prediction,test):
    TP=0 #真为真
    TN=0#假为假
    FN=0#假为真
    FP=0#真为假
    for item in range(len(test)):
        if (prediction[item]==1 and test[item]==1):
                TP=TP+1
        elif (prediction[item]==0 and test[item]==0):
                TN=TN+1
        elif (prediction[item] == 0 and test[item] == 1):
                FN=FN+1
        elif (prediction[item] == 1 and test[item]== 0):
                FP=FP+1
    return float((FN+FP)/len(test))
def opentxt(filename):
    a = list()
    fp = open(filename).readlines()
    for i in fp:
        c=i.split('\t')
        a.append([float(c[0]), float(c[1]),float(c[2])])
    return a
def sul(a,b,c,k):
    pre=[]
    spe=[]
    for i in c:
        p = 0  # 记录男生类别投票数
        q = 0  # 记录女生类别投票数
        i = numpy.array(i)
        z=k
        m = list()  # 存放与男生类的距离
        n = list()  # 存放与 女生类的距离
        for j in a:
            j = numpy.array(j)
            m.append(numpy.sum(numpy.square(i - j)))  # 男生样本点的距离
        for j in b:
            j = numpy.array(j)
            n.append(numpy.sum(numpy.square(i - j)))  # 女生样本点的距离
        while z != 0:  # 投票算法
            M = min(m)
            N = min(n)
            if M < N:
                p = p + 1
                m.remove(M)
            elif M>N:
                q = q + 1
                n.remove(N)
            z = z - 1
        if (p > q):
            pre.append(1)
        else:
            pre.append(0)
        if (p==q)or((p-q)==1)or((q-p)==-1):
            spe.append(list(i))
    return pre,spe
def main():
    a=opentxt("boy.txt")
    ans=list()#存放样本结果
    ax = plt.subplot(111, projection='3d')
    for i in a:
        ax.scatter(i[0],i[1],i[2],c='b')
    b=opentxt("girl.txt")
    for i in b:
        ax.scatter(i[0],i[1],i[2],c='r')
    c=opentxt("boy82.txt")
    d=opentxt("girl42.txt")
    for i in c:
        ans.append(1)
    for i in d:
        c.append(i)
        ans.append(0)
    y=[]
    x=[]
    for k in range(1,11,1):
        pre=sul(a,b,c,k)[0]
        y.append(ROC(pre,ans))
    k=3
    sur=sul(a,b,c,3)[1]
    sur=array(sur).T
    print(sur)
    ax.plot_trisurf(sur[0], sur[1], sur[2], linewidth=0.2, antialiased=True)
    for i in range(1,11,1):
        x.append(i)
    plt.show()
    plt.plot(x,y,'r')
    ans2=[]
    for i in a:
        ans2.append(1)
    for i in b:
        ans2.append(0)
    y=[]
    for k in range(1,11,1):
        pre=sul(a,b,a+b,k)[0]
        y.append(ROC(pre,ans2))
    plt.plot(x,y,'b')
    plt.show()
main()
