# 构建简单数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

# 将数据集从列表转换为字典
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

# FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    # 以文本形式显示树
    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

# 更新头指针表
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# FP树填充
def updateTree(items, inTree, headerTable, count):
    # 已存在该子节点，更新计数值
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    # 不存在该子节点，新建树节点
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 填充剩下的元素
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

# 使用数据集以及最小支持度，构建FP树
def createTree(dataSet, minSup=1):
    # 头指针表
    headerTable = {}
    # 首次遍历数据集，计数
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 删除非频繁项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # 根据全局频率对事务中的元素进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # FP树填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat,treeNode):
    condPats = {}
    while(treeNode != None):
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, hearderTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(hearderTable.items(), key=lambda x:x[1][0], reverse=True)]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, hearderTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            print('conditional tree for :', newFreqSet)
            myCondTree.disp()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    simData = loadSimpDat()
    # print(simData)
    initSet = createInitSet(simData)
    # print(initSet)
    myFPtree, myHeaderTable = createTree(initSet, 3)
    # myFPtree.disp()
    # print(myHeaderTable)
    test1 = findPrefixPath('x', myHeaderTable['x'][1])
    # print(test1)
    test2 = findPrefixPath('z', myHeaderTable['z'][1])
    # print(test2)
    test3 = findPrefixPath('r', myHeaderTable['r'][1])
    # print(test3)
    freqItem = []  # 频繁项集
    mineTree(myFPtree, myHeaderTable, 3, set([]), freqItem)