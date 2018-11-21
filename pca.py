from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename,delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

# PCA算法
def pca(dataMat,topNfeat = 9999999):
    # 求数据矩阵每一列的均值
    meanVals = mean(dataMat, axis=0)
    print(meanVals)
    # 数据矩阵每一列特征减去该列的特征均值
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

# 将NaN替换成平均值的函数
def replaceNaNWithMean():
    dataMat = loadDataSet('D:/PycharmProjects/Machine/machinelearninginaction/Ch13/secom.data', ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        # 利用该维度所有非NaN特征求取均值
        # nonzero返回数组中非零元素的索引值数组。
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat

# filePath  = 'D:/PycharmProjects/Machine/machinelearninginaction/Ch13/testSet.txt'
# dataMat = loadDataSet(filePath)
# lowDMat, reconMat = pca(dataMat, 2)
# print(shape(lowDMat))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker = '^',s = 90)
# ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker = 'o',s = 90,c = 'red')
# plt.show()

# 利用PCA对半导体制造数据降维
dataMat = replaceNaNWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals)