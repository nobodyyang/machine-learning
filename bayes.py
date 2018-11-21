# -*- coding: UTF-8 -*-
from numpy import *
import re
import feedparser


def loadDataSet():
    """
        Function:
            创建实验样本
        Parameters:
            无
        Returns:
            postingList - 实验样本切分的原始词条列表，列表每一行代表一个文档
            classVec - 类别标签向量
        Modify:
            2018-08-11
        """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 统计所有文档中出现的词条列表,即没有重复的单词
def createVocabList(dataSet):
    """
        Function:
            创建一个包含在所有文档中出现的不重复词的列表
        Parameters:
            dataSet - 样本切分词条数据集
        Returns:
            vocabSet - 返回不重复的词条列表，也就是词汇表
        Modify:
            2018-08-11
        """
    vocabSet = set([])
    for document in dataSet:
        # 将文档列表转为集合的形式，保证每个词条的唯一性
        # 然后与vocabSet取并集，向vocabSet中添加没有出现新的词条
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
        Function:
            根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量
        Parameters:
            vocabList - createVocabList返回的列表
            inputSet - 切分的词条列表
        Returns:
            returnVec - 文档向量,词集模型
        Modify:
            2018-08-11
        """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec


# 朴素贝叶斯分类器的训练函数
def trainNB0(trainMatrix, trainCategory):
    """
        Function:
            朴素贝叶斯分类器的训练函数
        Parameters:
            trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
            trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
        Returns:
            p0Vect - 非侮辱类的条件概率数组
            p1Vect - 侮辱类的条件概率数组
            pAbusive - 文档属于侮辱类的概率
        Modify:
            2018-08-11
        """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
            p1Num += trainMatrix[i]
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
        Function:
            朴素贝叶斯分类函数
        Parameters:
            vec2Classify - 待分类的词条数组
            p0Vec - 侮辱类的条件概率数组
            p1Vec -非侮辱类的条件概率数组
            pClass1 - 文档属于侮辱类的概率
        Returns:
            0 - 属于非侮辱类
            1 - 属于侮辱类
        Modify:
            2018-08-15
        """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
        Function:
            朴素贝叶斯分类测试函数
        Parameters:
            无
        Returns:
            无
        Modify:
            2018-08-15
        """
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, array(listClasses))
    # 测试文档
    testEntry = ['love', 'my', 'dalmation']
    # 将测试文档转为词条向量，并转为NumPy数组的形式
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    # 第二个测试文档
    testEntry1 = ['stupid', 'garbage']
    # 同样转为词条向量，并转为NumPy数组的形式
    thisDoc1 = array(setOfWords2Vec(myVocabList, testEntry1))
    print(testEntry1, 'classified as:', classifyNB(thisDoc1, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    """
        Function:
            朴素贝叶斯词袋模型，根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量
        Parameters:
            vocabList - createVocabList返回的列表
            inputSet - 切分的词条列表
        Returns:
            returnVec - 文档向量,词集模型
        Modify:
            2018-08-15
        """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec


def textParse(bigString):
    """
        Function:
            将大字符串其解析为字符串列表
        Parameters:
            bigString - 邮件字符串
        Returns:
            tok - 字符串列表
        Modify:
            2018-08-15
        """
    # 对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
    listOfTokens = re.split(r'\W*', bigString)
    # 将分割后的字符串中所有的大些字母变成小写lower(), 并且只保留单词长度大于3的单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    """
        Function:
            垃圾邮件测试函数
        Parameters:
            无
        Returns:
            无
        Modify:
            2018-08-15
        """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(
            open('D:/PycharmProjects/Machine/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(
            open('D:/PycharmProjects/Machine/machinelearninginaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 选10组做测试集，根据随机产生索引值获取
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 生成训练矩阵及标签
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 测试并计算错误率
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))


if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # for each in postingList:
    #     print(each)
    # print(classVec)

    # myVocabList = createVocabList(postingList)
    # print('词汇表：', myVocabList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print('词条向量:', trainMat)
    #
    # p0Vect, p1Vect, pAbusive = trainNB0(trainMat, classVec)
    # print('p0V:\n', p0Vect)
    # print('p1V:\n', p1Vect)
    # print('classVec:\n', classVec)
    # print('pAb:\n', pAbusive)

    # testingNB()

    spamTest()
    spamTest()
