import numpy as np
import matplotlib.pyplot as plt
def opentxt1(filename):
    a = list()
    fp = open(filename).readlines()
    for i in fp:
        a.append([float(i.split('\t')[0]), float(i.split('\t')[1]),float(i.split('\t')[2])])
    return a
def PCA(A,B): #求出来的投影方向
    B=B.T
    D=dict(zip(A,B))
    x=[]
    for i in range(0,2):
        x.append(D[A[i]])
    A=list(A)
    A=sorted(A,reverse=True)
    x=np.array(x).T
    return x
def pca():#样本处理
    a=opentxt1("boy.txt")#a为去中心化的数据样本集
    b=opentxt1("girl.txt")
    c=list()
    for i in a:
        c.append(i)
    for i in b:
        c.append(i)
    mean=np.mean(c,axis=0)
    a=a-mean
    b=b-mean
    a=np.vstack((a,b))
    a=np.matrix(a)
    S=np.cov(a.T)
    A, B = np.linalg.eig(S)#A为特征值，B为特征向量
    x=PCA(A,B)
    print(B,x)
    a=np.dot(a,x)
    b=np.dot(b,x)
    a=a.tolist()
    x=[a[i][0] for i in range(len(a))]
    y=[a[i][1] for i in range(len(a))]
    plt.plot(x,y,'o')
    x=[row[0] for row in b]
    y=[row[1] for row in b]
    plt.plot(x,y,'o')
    plt.show()
pca()

