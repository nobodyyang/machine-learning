#-*- coding: UTF-8 -*-
import operator
from math import log
from treePlotter import *
# 序列化模块
import pickle

# 建立测试calcShannonEnt()数据集
def creatDataSet():
    """
        Function:
            创建测试数据集
        Parameters:
            无
        Returns:
            无
        Modify:
            2018-08-02
        """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1 ,0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels =['no surfacing', 'flippers']
    #返回数据集和类标签
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 安照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 以下两行代表去除该行的featVec[axis]元素
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 选择最好的数据划分方式ID3
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 当遍历完所有的特征属性后，类标签仍然不唯一(分支下仍有不同分类的实例,采用多数表决的方法完成分类)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树函数
def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if (classList.count(classList[0]) == len(classList)):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatureLabel][value] = creatTree(splitDataSet(dataSet, bestFeature, value), subLables)
    return myTree

# 使用决策树的分类函数
def classify(inputTreee, featLabels, testVec):
    firstStr = list(inputTreee.keys())[0]
    secondDict = inputTreee[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLable = classify(secondDict[key], featLabels, testVec)
            else: classLable = secondDict[key]
    return classLable

# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# 获取保存好的决策树
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    # splitData = splitDataSet(dataSet, 0, 1)
    # dataSetEnt = calcShannonEnt(dataSet)
    # print(dataSetEnt)
    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print(bestFeature)
    myTree = creatTree(dataSet, labels)
    print(myTree)
    storeTree(myTree,'DecisionTree.txt')
    # myTree = retrieveTree(0)

    # print(myTree)
    # m = classify(myTree, labels, [1, 0])
    # print(m)