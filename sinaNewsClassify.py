import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def textProcessing(folderPath, testSize=0.2):
    """
        Function:
            中文文本处理
        Parameters:
            folderPath - 文本存放的路径
            testSize - 测试集占比，默认占所有数据集的百分之20
        Returns:
            allWordsList - 按词频降序排序的训练集列表
            trainDataList - 训练集列表
            testDataList - 测试集列表
            trainClassList - 训练集标签列表
            testClassList - 测试集标签列表
        Modify:
            2018-08-22
        """
    folderList = os.listdir(folderPath)
    dataList = []
    classList = []
    # for folder in folderList[0:1]:
    for folder in folderList:
        newFolderList = os.path.join(folderPath, folder)
        files = os.listdir(newFolderList)
        # print(files)
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(newFolderList, file), 'r', encoding='utf-8') as f:
                raw = f.read()
            # 精简模式，返回一个可迭代的generator
            wordCut = jieba.cut(raw, cut_all=False)
            # generator转换为list
            wordList = list(wordCut)
            # 添加数据集数据
            dataList.append(wordList)
            # 添加数据集类别
            classList.append(folder)
            j += 1
    # zip()将对象中对应的元素打包成一个个元组
    dataClassList = list(zip(dataList, classList))
    # 将data_class_list乱序
    random.shuffle(dataClassList)
    # 训练集和测试集切分的索引值
    index = int(len(dataClassList) * testSize) + 1
    trainList = dataClassList[index:]
    testList = dataClassList[:index]
    # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
    trainDataList, trainClassList = zip(*trainList)
    testDataList, testClassList = zip(*testList)

    allWordsDict = {}
    for wordList in trainDataList:
        for word in wordList:
            if word in allWordsDict.keys():
                allWordsDict[word] += 1
            else:
                allWordsDict[word] = 1

    # dict.items()函数以列表返回可遍历的(键, 值)元组数组
    # 根据键的值倒序排序
    allWordsTupleList = sorted(allWordsDict.items(), key=lambda f: f[1], reverse=True)
    allWordsList, allWordsNums = zip(*allWordsTupleList)
    allWordsList = list(allWordsList)
    return allWordsList, trainDataList, testDataList, trainClassList, testClassList


def makeWordsSet(wordsPath):
    """
        Function:
            读取文件里的内容，并去重
        Parameters:
            folderPath - 文件路径
        Returns:
            wordsSet - 读取的内容的set集合
        Modify:
            2018-08-22
        """
    wordsSet = set()
    with open(wordsPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                wordsSet.add(word)
    return wordsSet


def wordsDict(allWordsList, deleteN, stopWordsSet=set()):
    """
        Function:
            文本特征选取
        Parameters:
            allWordsList - 按词频降序排序的训练集所有文本列表
            deleteN - 删除词频最高的deleteN个词
        Returns:
            featureWords - 特征集
        Modify:
            2018-08-22
        """
    featureWords = []
    n = 1
    for t in range(deleteN, len(allWordsList), 1):
        if n > 1000:
            break
        if not allWordsList[t].isdigit() and allWordsList[t] not in stopWordsSet and 1 < len(allWordsList[t]) < 5:
            featureWords.append(allWordsList[t])
        n += 1
    return featureWords


def matrixFeatures(trainDataList, testDataList, featureWords):
    """
        Function:
            根据feature_words将文本向量化
        Parameters:
            trainDataList - 训练集
            testDataList - 测试集
            featureWords - 特征集
        Returns:
            trainFeatureList - 训练集向量化列表
            testFeatureList - 测试集向量化列表
        Modify:
            2018-08-22
        """

    def matrixFeature(text, featureWords):
        textWords = set(text)
        # 出现在特征集中，则置1
        features = [1 if word in textWords else 0 for word in featureWords]
        return features

    trainFeatureList = [matrixFeature(text, featureWords) for text in trainDataList]
    testFeatureList = [matrixFeature(text, featureWords) for text in testDataList]
    return trainFeatureList, testFeatureList


def sinaNewsClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList):
    classifier = MultinomialNB().fit(trainFeatureList, trainClassList)
    testAccuracy = classifier.score(testFeatureList, testClassList)
    return testAccuracy


if __name__ == '__main__':
    folderPath = './machinelearninginaction/Ch04/SogouC/Sample'
    # textProcessing(folderPath)

    allWordsList, trainDataList, testDataList, trainClassList, testClassList = textProcessing(folderPath)
    stopWordsFile = './machinelearninginaction/Ch04/SogouC/stopwords_cn.txt'
    stopWordsSet = makeWordsSet(stopWordsFile)
    # featureWords = wordsDict(allWordsList, 100, stopWordsSet)
    # print(featureWords)

    testAccuracyList = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        featureWords = wordsDict(allWordsList, deleteN, stopWordsSet)
        trainFeatureList, testFeatureList = matrixFeatures(trainDataList, testDataList, featureWords)
        testAccuracy = sinaNewsClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList)
        testAccuracyList.append(testAccuracy)

    plt.figure()
    plt.plot(deleteNs, testAccuracyList)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
