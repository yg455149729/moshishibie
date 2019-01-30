from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LDA
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def getdata(filename1,classnumber):
    '''
    导入单个txt文件，将其转成列表
    :param filename1: 文件名
    :param classnumber: 将其判为哪一类，0or1
    :return: 生成好的data,labels
    '''
    f  = open(filename1)
    lines  = f.readlines()
    mansls= []
    for line in lines:
        a=line.split()
        a= [float(i) for i in a ]
        mansls.append(a)

    manslabels= [classnumber]*shape(mansls)[0]
    return mansls,manslabels

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=99999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T) +meanVals
    return lowDDataMat,reconMat

def replaceNanwithMean():
    datMat = loadDataSet('boy.txt')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

mansls,manslabels1=getdata('boy.txt',0)
girlsls,girlslabels1=getdata('girl.txt',1)
m=mansls.copy()
g=girlsls.copy()
ml=manslabels1.copy()
gl=girlslabels1.copy()
ml.extend(gl)
alllabels = ml
m.extend(g)
alldatas = mat(m)
# print(array(mansls)[:, 0])
#


#
# plt.show()
dataMat=alldatas
# dataMat = replaceNanwithMean()
# print(dataMat)
meanVals = mean(dataMat,axis =0)
# print(meanVals)
meanRemoved = dataMat - meanVals
# #print(meanRemoved)
covMat = cov(meanRemoved,rowvar=0)
# #print(covMat)

eigvals,eigVects = linalg.eig(mat(covMat))
w=eigvals[:2]
# eigVects[:2]=eigVects[:2]+mat([[1000,100,100],[100,100,100]])
down_Mat=dot(dataMat,eigVects[0:2].T)#降维后的数据
eigVects[:2]=eigVects[:2]*300
# print(eigVects[:2])

ax=plt.subplot(221,projection='3d')
plt.title('均一化')
ax.scatter(array((mansls-mean(mansls))/sum(mansls))[:, 0], array((mansls-mean(mansls))/sum(mansls))[:, 1], array((mansls-mean(mansls))/sum(mansls))[:, 2], c='red')
ax.scatter(array((girlsls-mean(girlsls))/sum(girlsls))[:, 0], array((girlsls-mean(girlsls))/sum(girlsls))[:, 1], array((girlsls-mean(girlsls))/sum(girlsls))[:, 2], c='green')

ax=plt.subplot(223,projection='3d')
plt.title('原始数据')
ax.scatter(array(mansls)[:, 0], array(mansls)[:, 1], array(mansls)[:, 2], c='red')
ax.scatter(array(girlsls)[:, 0], array(girlsls)[:, 1], array(girlsls)[:, 2], c='green')

ax=plt.subplot(222,projection='3d')
plt.title('投影平面')
plt.xlim(150,200)
plt.ylim(40,90)

ax.scatter(array(mansls)[:, 0], array(mansls)[:, 1], array(mansls)[:, 2], c='red')
ax.scatter(array(girlsls)[:, 0], array(girlsls)[:, 1], array(girlsls)[:, 2], c='green')
ax.plot_trisurf([0,eigVects[:2][0,0],eigVects[:2][1,0]], [0,eigVects[:2][0,1],eigVects[:2][1,1]],[0,eigVects[:2][0,2],eigVects[:2][1,2]])



# print(array(down_Mat)[:len(mansls), 0])
plt.subplot(224)
plt.title('投影之后')
plt.scatter(array(down_Mat)[:len(mansls), 0],array(down_Mat)[:len(mansls), 1], c='red')
plt.scatter(array(down_Mat)[len(mansls):, 0], array(down_Mat)[len(mansls):, 1], c='green')

plt.show()

# a,l=LDA.getdata_girl_and_man('boy.txt','girl.txt')

#用降维的数据测试LDA
w,mean1,mean2,group1,group2= LDA.train(down_Mat[:len(mansls), :],down_Mat[len(mansls):, :])
# drawcompare(w,mean1,mean2,group1,group2,groupboy,groupgirl)//贝叶斯和LDA决策面
testboy,labels=LDA.getdata('boy82.txt',0)
#
group3=mat(np.array(testboy)[:,0:2].T)
# print(group3[:,1].T)
# draw(w,mean1,mean2,group3,group2)
count=0
print('1:',(LDA.predict(group3[:,1].T,w,mean1,mean2)))
for i in range(shape(group3)[1]):
    if (LDA.predict(group3[:,i].T,w,mean1,mean2))[0,0]>=0:#这里要注意和测试数据对应
        count = count+1
print(count/shape(group3)[1])#降维后有0.86准确率，而没降维只有0.84