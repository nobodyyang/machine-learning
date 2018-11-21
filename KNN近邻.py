#-*- coding: utf-8 -*-
from numpy import *
from numpy import array
# 运算法包
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables

# group, lables = creatDataSet()
# print(group)
# print(lables)

def classify0(inX, dataSet , lables, k):
    dataSetSize = dataSet.shape[0]
    # tile(A, reps)将A进行重复输出
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqrt(sqDistance)
    # argsort()返回数组值从小到大的索引值
    sortedDistIndicies = distance.argsort()
    classCount = {}

    for i in range(k):
        voteIlable = lables[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1

    # dict.items()以列表返回可遍历的(键, 值) 元组数组
    # 使用字典的第二个元素进行降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 讲文本记录转换为Numpy
def file2Matrix(filename):
    f = open(filename, 'r')
    arrayOlines = f.readlines()
    numberOlines = len(arrayOlines)
    returntMat = zeros((numberOlines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        listFromLine = line.strip().split('\t')
        returntMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return  returntMat,classLabelVector

def autoNorm(dataSet):
    # 获取数据集中每一列的最小数值
    minVals = dataSet.min(0)
    # 获取数据集中每一列的最大数值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = shape(dataSet)[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2Matrix( 'C:/Users/Administrator/Desktop/机器学习实战源码/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d, result is :%s" % (
         classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    # 定义预测结果
    resultList = ['not at all', 'in small does', 'in large does']
    # input()接收任意任性输入，将所有输入默认为字符串处理，并返回字符串类型。
    percentTats = float(input( "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    # 将输入的数值放在数组中
    inArr = array([ffMiles, percentTats, iceCream])
    datingDataMat, datingLabels = file2Matrix(
        'C:/Users/Administrator/Desktop/机器学习实战源码/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    classifierResult = classify0((inArr - minValues) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person:", resultList[classifierResult - 1])

# 将图像转化未测试向量
def img2Vector(filename):

    returnVect = zeros((1, 1024))
    f = open(filename, 'r')
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0,32 * i + j] = int(lineStr[j])
    return returnVect

# 手写数字识别系统
def handwritingClassTest():
    hwLabels = []
    trainingFileDir = 'D:/PycharmProjects/Machine/machinelearninginaction/Ch02/digits/trainingDigits'
    testFileDir = 'D:/PycharmProjects/Machine/machinelearninginaction/Ch02/digits/testDigits'
    trainingFileList = listdir(trainingFileDir)
    m = len(trainingFileList)
    print('training m is %d'%m)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector(trainingFileDir + '/' + fileNameStr)

    testFileList = listdir(testFileDir)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector(testFileDir + '/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifierResult came back with: %d,the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr ): errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    # dataSet, labels = creatDataSet()
    # inX = [0, 0]
    # a = classify0(inX, dataSet, lables, 3)
    # print(a)
    # datingDataMat,datingLabels = file2Matrix('C:/Users/Administrator/Desktop/机器学习实战源码/machinelearninginaction/Ch02/datingTestSet2.txt')
    # 使用Matplotlib创建散点图
    # fig = plt.figure()
    # add_subplot(3位数字参数)子图总行数,子图总列数,子图位置
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()
    # normMat, ranges ,minVals = autoNorm(datingDataMat)
    # classifyPerson()
    handwritingClassTest()