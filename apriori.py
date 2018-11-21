def loadDataSet():
    return [[1,3,4], [2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)

def scanD(D, CK, minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if not (can in ssCnt): ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def apriorGen(LK, k):
    retList = []
    lenLk = len(LK)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList


def apriori(dataSet, minSupport = 0.5):
    C1 = list(createC1(dataSet))
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = apriorGen(L[k-2], k)
        Lk,supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

def calcConf(freqSet, H, supportData, brl, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq,conf))
            prunedH.append(conseq)
    return prunedH


def rulesFormConseq(freqSet, H, supportData, brl, minConf):
    m = len(H[0])
    if (len(freqSet) > m + 1):
        Hmp1 = apriorGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFormConseq(freqSet, Hmp1, supportData, brl, minConf)

def generateRules(L, supportData, minConf = 0.7):
    bigRulesList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFormConseq(freqSet, H1, supportData, bigRulesList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRulesList, minConf)
    return bigRulesList

if __name__ == '__main__':
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet, 0.5)
    print('频繁项集L：',L)
    print('所有候选项集的支持度信息：',supportData)
    rules = generateRules(L, supportData, 0.5)
    print('rules:',rules)
    # C1 = list(createC1(dataSet))
    # D = list(map(set,dataSet))
    # print(D)
    # L1, suppData = scanD(D, C1, 0.1)
    # print(L1)
    # print(suppData)