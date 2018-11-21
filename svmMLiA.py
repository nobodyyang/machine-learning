from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    """
        Function:
            读取数据
        Parameters:
            fileName - 文件名
        Returns:
            dataMat - 数据矩阵
            labelMat - 数据标签
        Modify:
            2018-09-21
    """
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
        Function:
            随机选择alphaJ
        Parameters:
            i - alpha下标
            m - alpha参数个数
        Returns:
            j - 随机选择alpha另一个下标
        Modify:
            2018-09-21
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """
        Function:
            修剪alpha
        Parameters:
            aj - alpha值
            H - alpha上限
            L - alpha下限
        Returns:
            aj - alpah值
        Modify:
            2018-09-21
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
        Function:
            简化版SMO算法
        Parameters:
            dataMatIn - 数据矩阵
            classLabels - 数据标签
            C - 松弛变量
            toler - 容错率
            maxIter - 最大迭代次数
        Returns:
            无
        Modify:
            2018-09-21
    """
    # 转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn)
    # 类别标签列向量
    labelMat = np.mat(classLabels).transpose()
    # 初始化b参数，统计dataMatrix的维度
    b = 0
    m, n = np.shape(dataMatrix)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代matIter次
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率。
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] *\
                            dataMatrix[i, :].T - dataMatrix[j, ] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 步骤5：修剪alpha_j
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('alpha_j变化太小')
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T -\
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T -\
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("第%d次迭代样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
        # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas


def show_classifier(dataMat, class_labels,w, b, alphas):
    """
        Function:
            可视化分类结果
        Parameters:
            dataMat - 数据矩阵
            w - 直线法向量
            b - 直线截距
            alphas - alphas值
        Returns:
            无
        Modify:
            2018-09-22
    """
    # 绘制样本点
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if class_labels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


def calc_ws(alphas, data_arr, class_labels):
    """
        Function:
            计算w
        Parameters:
            data_arr - 数据矩阵
            class_labels - 数据标签
            alphas - alphas值
        Returns:
            w - 计算得到的w
        Modify:
            2018-09-22
    """
    X = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w


class OptStruct(object):
    def __init__(self, dataMatIn, classLabels, C, toler):
        """
            Parameters:
                dataMatIn - 数据矩阵
                classLabels - 数据标签
                C - 松弛变量
                toler - 容错率
        """
        self.X = dataMatIn
        self.label_mat = classLabels
        self.C = C
        self.tol = toler
        # 数据矩阵行数
        self.m = np.shape(dataMatIn)[0]
        # 根据矩阵行数初始化alpha参数为0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calc_Ek(oS, k):
    """
        Function:
            计算误差
        Parameters:
            oS - 数据结构
            k - 下标为k的数据
        Returns:
             Ek - 下标为k的数据误差
        Modify:
            2018-09-22
    """
    fXk = float(np.multiply(oS.alphas, oS.label_mat).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fXk - float(oS.label_mat[k])
    return Ek


def select_J(i, oS, Ei):
    """
        Function:
            内循环启发方式
        Parameters:
            i - 下标为i的数据的索引值
            oS - 数据结构
            Ei - 下标为i的数据误差
        Returns:
            j, maxK - 下标为j或maxK的数据的索引值
            Ej - 下标为j的数据误差
        Modify:
            2018-09-22
    """
    maxK = -1
    max_deltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    # 返回Ei不为0的索引值数组
    valid_eCache_list = nonzero(oS.eCache[:, 0].A)[0]
    # 有不为0的Ei
    if (len(valid_eCache_list)) > 1:
        for k in valid_eCache_list:
            if k == i: continue
            Ek = calc_Ek(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > max_deltaE):
                maxK = k
                max_deltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 没有不为0的Ei
    else:
        j = selectJrand(i, oS.m)
        Ej = calc_Ek(oS, j)
    return j, Ej


def update_Ek(oS, k):
    """
        Function:
            计算Ek,并更新误差缓存
        Parameters:
            oS - 数据结构
            k - 下标为k的数据的索引值
        Returns:
            无
        Modify:
            2018-09-22
    """
    Ek = calc_Ek(oS, k)
    # 更新Ei缓存
    oS.eCache[k] = [1, Ek]


def inner_L(i, oS):
    """
        Function:
            优化的SMO算法
        Parameters:
            i - 下标为i的数据的索引值
            oS - 数据结构
        Returns:
            1 - 有任意一对alpha值发生变化
            0 - 没有任意一对alpha值发生变化或变化太小
        Modify:
            2018-09-22
    """
    # 步骤1：计算误差Ei
    Ei = calc_Ek(oS, i)
    if ((oS.label_mat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.label_mat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = select_J(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alpha_i_old = oS.alphas[i].copy()
        alpha_j_old = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if (oS.label_mat[i] != oS.label_mat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
            # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.label_mat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        # 更新Ei至误差缓存
        update_Ek(oS, j)
        if (abs(oS.alphas[j] - alpha_j_old) < 0.00001):
            print('alpha_j变化太小')
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.label_mat[j] * oS.label_mat[i] * (alpha_j_old - oS.alphas[j])
        # 步骤7：更新b1和b2
        b1 = oS.b - Ei - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[i, :].T - \
             oS.label_mat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[j, :].T - \
             oS.label_mat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b1和b2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(data_arr, class_labels, C, toler, max_iter):
    """
    Function:
        完整的线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    Modify:
        2018-09-23
    """
    oS = OptStruct(np.mat(data_arr), np.mat(class_labels).transpose(), C, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    # 遍历整个数据集alpha也都没有更新或者超过最大迭代次数则退出循环
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        # 遍历整个数据集
        if entire_set:
            for i in range(oS.m):
                # 使用优化的SMO算法
                alpha_pairs_changed += inner_L(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_L(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历一次后改为非边界遍历
        if entire_set:
            entire_set = False
        # 如果alpha没有更新,计算全样本遍历
        elif (alpha_pairs_changed == 0):
            entire_set = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas


def calc_ws(alphas, data_arr, class_labels):
    """
        Function:
            计算w
        Parameters:
            data_arr - 数据矩阵
            class_labels - 数据标签
            alphas - alphas值
        Returns:
            w - 计算得到的w
        Modify:
            2018-09-22
    """
    X = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w


def show_data_set(data_mat, label_mat):
    """
        Function:
            可视化数据集
        Parameters:
            data_mat - 数据矩阵
            label_mat - 数据标签
        Returns:
            无
        Modify:
            2018-09-23
    """
    data_plus = []
    data_minus = []
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


if __name__ == '__main__':
    data_arr, class_labels = loadDataSet('./machinelearninginaction/Ch06/testSet.txt')
    # b, alphas = smoSimple(data_arr, class_labels, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas > 0])
    #
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(data_arr[i], class_labels[i])
    #
    # w = calc_ws(alphas, data_arr, class_labels)
    # show_classifier(data_arr, class_labels, w, b, alphas)
    #
    b, alphas = smoP(data_arr, class_labels, 0.6, 0.001, 40)
    w = calc_ws(alphas, data_arr, class_labels)
    show_classifier(data_arr ,class_labels,w, b,alphas)