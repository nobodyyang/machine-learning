# -*- coding: utf-8 -*-
from cmath import exp
from numpy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def loadDataSet():
    """
    Function:
        加载数据集
    Parameters:
        无
    Returns:
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-08-27
    """
    dataMat = []
    labelMat = []
    fr = open('./machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


def plot_data_set():
    """
    Function:
        绘制数据集
    Parameters:
        无
    Returns:
        无
    Modify:
        2018-08-27
    """
    data_mat, label_mat = loadDataSet()
    data_arr = array(data_mat)
    n = shape(data_mat)[0]
    xcord_1 = []
    ycord_1 = []
    xcord_2 = []
    ycord_2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord_1.append(data_arr[i, 1])
            ycord_1.append(data_arr[i, 2])
        else:
            xcord_2.append(data_arr[i, 1])
            ycord_2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord_1, ycord_1, s=20, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord_2, ycord_2, s=20, c='green', marker='s', alpha=0.5)
    plt.title('data_set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def sigmoid(inX):
    """
    Function:
        Sigmoid函数
    Parameters:
        inX - 数据
    Returns:
        Sigmoid函数
    Modify:
        2018-08-27
    """
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    Function:
        梯度上升算法
    Parameters:
        dataMatIn - 数据集
        classLabels - 类别标签
    Returns:
        weights - 求得的权重，最优参数
    Modify:
        2018-08-27
    """
    # dataMatrix 100 * 3
    dataMatrix = mat(dataMatIn)
    # labMatrix 100 * 1
    labMatrix = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        # h 100 * 1
        h = sigmoid(dataMatrix * weights)
        error = (labMatrix - h)
        # weights 3 * 1
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()


def poltBestFit(weights):
    """
    Function:
        画出数据集和Logistic回归最佳拟合直线函数
    Parameters:
        dataMatIn - 数据集
        classLabels - 类别标签
    Returns:
        weights - 求得的权重，最优参数
    Modify:
        2018-08-28
    """
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-4.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabel):
    """
    Function:
        随机梯度上升算法
    Parameters:
        weights - 最优参数
    Returns:
        无
    Modify:
        2018-08-27
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # h 1 * 1
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabel[i] - h
        # weights 3 * 1
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMat, classLabel, numIter=150):
    """
    Function:
        改进的随机梯度上升算法
    Parameters:
        weights - 最优参数
    Returns:
        无
    Modify:
        2018-08-28
    """
    m, n = shape(dataMat)
    weights = ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    """
    Function:
        Logistic回归分类函数
    Parameters:
        inX - 特征向量
        weights - 回归系数
    Returns:
        分类结果
    Modify:
        2018-08-29
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    Function:
        Logistic回归分类测试函数
    Parameters:
        无
    Returns:
        分类结果错误率
    Modify:
        2018-08-29
    """
    frTrain = open('./machinelearninginaction/Ch05/horseColicTraining.txt')
    frTest = open('./machinelearninginaction/Ch05/horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用改进的随机梯度上升算法
    # trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # 使用梯度上升算法
    trainWeights = gradAscent(array(trainingSet), trainingLabels)
    errorCount = 0.0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 使用改进的随机梯度上升算法
        # if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
        # 使用梯度上升算法
        if int(classifyVector(array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is: %f' % errorRate)
    return errorRate


def multiTest():
    """
    Function:
        多次调用colicTest()函数，求结果的平均值
    Parameters:
        无
    Returns:
        分类结果平均错误率
    Modify:
        2018-08-29
    """
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


def colic_sklearn():
    fr_train = open('./machinelearninginaction/Ch05/horseColicTraining.txt')
    fr_test = open('./machinelearninginaction/Ch05/horseColicTest.txt')
    training_set = []
    training_labels = []
    test_set = []
    test_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(len(curr_line) - 1):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[-1]))
    for line in fr_test.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(len(curr_line) - 1):
            line_arr.append(float(curr_line[i]))
            test_set.append(line_arr)
            test_labels.append(float(curr_line[-1]))
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(training_set, training_labels)
    test_accuracy = classifier.score(test_set, test_labels) * 100
    print('正确率：%f%%' % test_accuracy)


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # print(dataArr)
    # print(labelMat)
    # plot_data_set()

    # print(gradAscent(dataArr, labelMat))

    # weights = gradAscent(dataArr, labelMat)
    # poltBestFit(weights)

    # weights = stocGradAscent0(array(dataArr), labelMat)
    # poltBestFit(weights)

    # weights = stocGradAscent1(array(dataArr), labelMat, 500)
    # poltBestFit(weights)

    # multiTest()

    colic_sklearn()
