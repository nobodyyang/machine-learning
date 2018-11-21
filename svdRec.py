from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

Data = loadExData()
U, sigma, VT =  linalg.svd(Data)
# print(sigma)


Sig3 = mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
a = U[:, :3] * Sig3 * VT[:3, :]
# print(a)

# 欧式距离相似度计算
def euclidSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

# 皮尔逊相关系数相似度计算
def pearsSim(inA,inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

# 余弦距离相似度计算
def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

myMat = mat(loadExData())
print(euclidSim(myMat[:, 0], myMat[:, 4]))
print(euclidSim(myMat[:, 0], myMat[:, 0]))
print(pearsSim(myMat[:, 0], myMat[:, 4]))
print(pearsSim(myMat[:, 0], myMat[:, 0]))
print(cosSim(myMat[:, 0], myMat[:, 4]))
print(cosSim(myMat[:, 0], myMat[:, 0]))

def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRatin = dataMat[user, j]
        if userRatin == 0: continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]