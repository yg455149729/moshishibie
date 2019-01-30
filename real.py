import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import multivariate_normal

def opentxt(filename):#读取文件
    fp=open(filename,'r')
    a=fp.readlines()
    return a
def div(a):#把信息存放在一个列表中，，第一个元素存放身高，第二个存放体重，第三个鞋码
    c1=list()
    d1=list()
    e1=list()
    for i in a:
        c1.append(float(i.split('\t')[0]))
        d1.append(float(i.split('\t')[1]))
        e1.append(float(i.split('\t')[2]))
    return [c1,e1,d1]
def mean(num):#求平均值
    height=sum(num)/(len(num))

    return height
def standard(num):#求出三个参数的标准差
    avg=mean(num)
    i=0
    for item in num:
        i=pow(item-avg,2)+i
    dev=float(i/len(num))
    stdev= np.math.sqrt(dev)

    return stdev

def f1(x,num):  # 分别求三个的正态分布概率密度函数值
    prob1 = stats.norm.pdf(x, mean(num[0]), standard(num[0]))
    prob2 = stats.norm.pdf(x, mean(num[1]), standard(num[1]))
    prob3 = stats.norm.pdf(x, mean(num[2]), standard(num[2]))
    return [prob1, prob2, prob3]
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
    return float(TP/(TP+FN)),float(TN/(FP+TN))
    #第一个参数灵敏度，第二个为特异度

def canshu(a):
    p=list()
    for i in a:
        p.append([float(i.split("\t")[0]),float(i.split('\t')[1])])
    q=np.array(p)
    mu = np.mean(q, axis=0)
    s_in = 0
    num=q.shape[0]
    #for i in range(num):
       # x=q[i]-mu
        #s_in += np.dot(x, x.T)
    #thu = s_in / num
    thu=np.cov(q[:,0],q[:,1])
    return mu, thu
def f2(x,a):
    mu,thu=canshu(a)
    return multivariate_normal.pdf(x, mean=np.array(mu), cov=thu)

def main1():
    #学习
    a=div(opentxt("boy.txt"))
    b=div(opentxt("girl.txt"))
    #身高
    for item in [0,1,2]:
        p=list()
        q=list()
        for i in a[item]:
            j=f1(i,a)
            z=f1(i,b)
            p.append(j[item]*0.5/(j[item]*0.5+z[item]*0.5))
        for i in b[item]:
            q.append(f1(i,b)[item]*0.5/(f1(i,a)[item]*0.5+f1(i,b)[item]*0.5))
        p=list(set(p))
        q=list(set(q))
        x=list(set(a[item]))
        x.sort()
        plt.plot(x, p, 'r-o', label='$line$', linewidth=1)
        x=list(set(b[item]))
        x.sort()
        plt.plot(x, q, 'b-o', label='$line$', linewidth=1)
        plt.show()
    t=div(opentxt("boy82.txt"))


    ans=list()
    j=len(t[0])
    for i in range(j):
        ans.append(1)
    for i in range(3):
        t[i]=t[i]+div(opentxt('girl42.txt'))[i]
    for i in range(len(t[0])-j):
        ans.append(0)

    for j in range(2):
        x = list()
        y = list()
        for item in np.arange(0,1,0.01):
            prediction=list()
            for i in t[j]:
                p=f1(i,z)
                q=f1(i,b)
                if (p[j]*0.5/(p[j]*0.5+q[j]*0.5)>item):
                    prediction.append(1)
                else:
                    prediction.append(0)
            r1,r2=ROC(prediction,ans)
            x.append(r1)
            y.append(1-r2)
        plt.xlim(0,1)
        plt.ylim(0,1)
        if(j==0):
            plt.plot(y,x,'r', label='$line$', linewidth=1)
        elif(j==1):
            plt.plot(y, x, 'b', label='$line$', linewidth=1)
        elif(j==2):
            plt.plot(y, x, 'g', label='$line$', linewidth=1)
        plt.show()
    return 0
def main2():
    # 学习
    a = opentxt("boy.txt")
    b = opentxt("girl.txt")
    c=opentxt("boy82.txt")
    #训练散点图
    x=div(a)
    y=div(b)
    plt.plot(x[0],x[1],'o')
    plt.plot(y[0],y[1],'o')
    plt.show()

    j = list()
    for i in c:
        j.append([float(i.split("\t")[0]), float(i.split('\t')[1])])
    ans=list()
    for  i in range(len(j)):
        ans.append(1)
    d=opentxt("girl42.txt")
    x=list()
    y=list()
    for i in d:
        j.append([float(i.split("\t")[0]), float(i.split('\t')[1])])
    for i in range(len(j)-len(ans)):
        ans.append(0)
    o=list()
    print(j)
    for i in j:
        p=f2(i,a)
        q=f2(i,b)
        o.append(p*0.5/(q*0.5+p*0.5))
    print(max(o),min(o))
    print(j)
    for item in np.arange(min(o), max(o), 0.01):
        prediction=list()
        for i in j:
            p=f2(i,a)
            q=f2(i,b)
            if((p*0.5/(q*0.5+p*0.5))>item):
                prediction.append(1)
            else:
                prediction.append(0)
        r1, r2 = ROC(prediction, ans)
        x.append(r1)
        y.append(1-r2)
    plt.plot(y, x, 'r', label='$line$', linewidth=1)
    plt.show()
    return 0
main1()