# -*- coding: UTF-8 -*-
from math import log
import operator
import matplotlib.pyplot as plt
import pickle


def creatDataSet():
    """
        Function:
            创建测试数据集
        Parameters:
            无
        Returns:
            dataSet - 数据集
            labels - 分类属性标签
        Modify:
            2018-08-02
        """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
        Function:
            计算给定数据集经验熵
        Parameters:
            dataSet - 数据集
        Returns:
            shannonEnt - 经验熵
        Modify:
            2018-08-02
        """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
        Function:
            按照给定特征划分数据集
        Parameters:
            dataSet - 待划分的数据集
            axis - 划分数据集的特征
            value - 特征的取值
        Returns:
            retDataSet - 划分后的数据集
        Modify:
            2018-08-02
        """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 以下两行代表去除该行的featVec[axis]元素
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 选择最好的数据划分方式ID3
def chooseBestFeatureToSplit(dataSet):
    """
        Function:
            选择最优特征划分方式（计算信息增益）
        Parameters:
            dataSet - 数据集
        Returns:
            bestFeature - 信息增益最大的特征的索引值
        Modify:
            2018-08-02
        """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取特征i的特征值列表
        featList = [example[i] for example in dataSet]
        # 利用set集合元素唯一性的性质，得到特征i的取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算第i特征划分信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print('第%d个特征的增益为%.3f' % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
        Function:
            多数表决的方法完成分类
        Parameters:
            classList - 类标签列表
        Returns:
            sortedClassCount[0][0] - 出现次数最多的类标签
        Modify:
            2018-08-02
        """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
        Function:
            创建决策树
        Parameters:
            dataSet - 数据集
            labels - 分类属性标签
        Returns:
            myTree - 决策树
        Modify:
            2018-08-02
        """
    classList = [example[-1] for example in dataSet]
    # 判断所有类标签是否相同，相同则返回该类标签
    if (classList.count(classList[0]) == len(classList)):
        return classList[0]
    # 遍历完所有的特征属性，此时数据集的列为1，即只有类标签列
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优特征
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    # 采用字典嵌套字典的方式，存储分类树信息
    myTree = {bestFeatureLabel: {}}
    # 复制当前特征标签列表，防止改变原始列表的内容
    subLabels = labels[:]
    del (subLabels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


def getNumLeafs(myTree):
    """
        Function:
            获取叶节点的数目
        Parameters:
            myTree - 决策树
        Returns:
            numLeafs - 叶节点的数目
        Modify:
            2018-08-04
        """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
        Function:
            获取树的层数
        Parameters:
            myTree - 决策树
        Returns:
            numLeafs - 树的层数
        Modify:
            2018-08-04
        """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
        Function:
            绘制带箭头的注释
        Parameters:
            nodeTxt - 结点名
            centerPt - 文本位置
            parentPt - 标注的箭头位置
            nodeType - 结点格式
        Returns:
            无
        Modify:
            2018-08-04
        """
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 绘制结点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va="center",
                            ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    """
        Function:
            计算父节点和子节点的中间位置，并在此处添加简单的文本标签信息
        Parameters:
            cntrPt、parentPt - 用于计算标注位置
            txtString - 标注的内容
        Returns:
            无
        Modify:
            2018-08-04
        """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 计算宽与高
def plotTree(myTree, parentPt, nodeTxt):
    """
        Function:
            绘制决策树
        Parameters:
            myTree - 字典决策树
            parentPt - 标注的内容
            nodeTxt - 结点名
        Returns:
            无
        Modify:
            2018-08-04
        """
    # 定义文本框和箭头格式
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 标记子节点属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
        Function:
            绘树主函数
        Parameters:
            inTree - 字典决策树
        Returns:
            无
        Modify:
            2018-08-04
        """
    # 创建fig
    fig = plt.figure(1, facecolor='white')
    # 清空fig
    fig.clf()
    # 设置坐标轴数据
    axprops = dict(xticks=[], yticks=[])
    # 去除坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 两个全局变量plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置，
    # 以及放置下一个节点的恰当位置
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def classify(inputTreee, featLabels, testVec):
    """
        Function:
            使用决策树分类
        Parameters:
            inputTree - 训练好的决策树信息
            featLabels - 标签列表
            testVec - 测试向量
        Returns:
            无
        Modify:
            2018-08-04
        """
    # 获取决策树结点
    firstStr = list(inputTreee.keys())[0]
    # 下一个字典
    secondDict = inputTreee[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLable = classify(secondDict[key], featLabels, testVec)
            else:
                classLable = secondDict[key]
    return classLable


def storeTree(inputTree, filename):
    """
        Function:
            使用pickle模块存储决策树
        Parameters:
            inputTree - 已经生成的决策树
            filename - 决策树的存储文件名
        Returns:
            无
        Modify:
            2018-08-04
        """
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
        Function:
            获取保存好的决策树
        Parameters:
            filename - 决策树的存储文件名
        Returns:
            无
        Modify:
            2018-08-04
        """
    fr = open(filename, 'rb')
    return pickle.load(fr)


def predictLensesType(filename):
    """
        Function:
            使用决策树预测隐形眼镜类型
        Parameters:
            filename - 隐形眼镜数据集文件名
        Returns:
            无
        Modify:
            2018-08-04
        """
    # 打开文本数据
    fr = open(filename)
    # 将文本数据的每一个数据行按照tab键分割，并依次存入lenses
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 创建并存入特征标签列表
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 根据继续文件得到的数据集和特征标签列表创建决策树
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree


if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    dataSetEnt = calcShannonEnt(dataSet)
    # print(dataSetEnt)

    # retDataSet = splitDataSet(dataSet, 0, 0)
    # print(retDataSet)
    #
    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print('最优特征索引值:', bestFeature)

    # myTree = createTree(dataSet, labels)
    # print(myTree)

    # myTree = createTree(dataSet, labels)
    # print(myTree)
    # createPlot(myTree)

    # myTree = createTree(dataSet, labels)
    # classifyResult1 = classify(myTree, labels, [1, 0])
    # print(classifyResult1)
    # classifyResult2 = classify(myTree, labels, [1, 1])
    # print(classifyResult2)

    # myTree = createTree(dataSet, labels)
    # print(myTree)
    # storeTree(myTree, 'classifierStorage.txt')
    # impTree = grabTree('classifierStorage.txt')
    # print('impTree:', impTree)
    # classifyResult2 = classify(myTree, labels, [1, 1])
    # print('[1, 1]的分类为：', classifyResult2)

    lensesTree = predictLensesType('D:/PycharmProjects/Machine/machinelearninginaction/Ch03/lenses.txt')
    print(lensesTree)
    createPlot(lensesTree)