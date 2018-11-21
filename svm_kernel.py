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


def calc_Ek_kernel(oS, k):
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
    fXk = float(np.multiply(oS.alphas, oS.label_mat).T * (oS.K[:, k]) + oS.b)
    Ek = fXk - float(oS.label_mat[k])
    return Ek


def select_J_kernel(i, oS, Ei):
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
            Ek = calc_Ek_kernel(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > max_deltaE):
                maxK = k
                max_deltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 没有不为0的Ei
    else:
        j = selectJrand(i, oS.m)
        Ej = calc_Ek_kernel(oS, j)
    return j, Ej


def update_Ek_kernel(oS, k):
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
    Ek = calc_Ek_kernel(oS, k)
    # 更新Ei缓存
    oS.eCache[k] = [1, Ek]


def inner_l_kernel(i, oS):
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
    Ei = calc_Ek_kernel(oS, i)
    if ((oS.label_mat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.label_mat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = select_J_kernel(i, oS, Ei)
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
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.label_mat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        # 更新Ei至误差缓存
        update_Ek_kernel(oS, j)
        if (abs(oS.alphas[j] - alpha_j_old) < 0.00001):
            print('alpha_j变化太小')
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.label_mat[j] * oS.label_mat[i] * (alpha_j_old - oS.alphas[j])
        # 步骤7：更新b1和b2
        b1 = oS.b - Ei - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, i] - \
             oS.label_mat[j] * (oS.alphas[j] - alpha_j_old) * oS.K[i, j]
        b2 = oS.b - Ej - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, j] - \
             oS.label_mat[j] * (oS.alphas[j] - alpha_j_old) * oS.K[j, j]
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


def smoP_kernel(data_arr, class_labels, C, toler, max_iter, k_tup=('lin', 0)):
    """
    Function:
        完整的线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
        kTup - 包含核函数信息的元组
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    Modify:
        2018-09-23
    """
    oS = OptStructKernel(np.mat(data_arr), np.mat(class_labels).transpose(), C, toler, k_tup)
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
                alpha_pairs_changed += inner_l_kernel(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l_kernel(i, oS)
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


def kernel_trans(X, A, k_tup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    # 线性核函数,只进行内积
    if k_tup[0] == 'lin': K = X * A.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K / (-1 * k_tup[1] ** 2))
    else: raise NameError('核函数无法识别')
    return K



class OptStructKernel(object):
    def __init__(self, data_mat_in, class_labels, C, toler, k_tup):
        self.X = data_mat_in
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        # 数据矩阵行数
        self.m = np.shape(data_mat_in)[0]
        # 根据矩阵行数初始化alpha参数为0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 初始化核K
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 计算所有数据的核K
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


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

def test_rbf(k1=1.3):
    """
        Function:
            利用核函数进行分类的径向基测试函数
        Parameters:
            k1 - 使用高斯核函数的时候表示到达率
        Returns:
            无
        Modify:
            2018-09-26
    """
    data_arr, class_labels = loadDataSet('machinelearninginaction/Ch06/testSetRBF.txt')
    b, alphas = smoP_kernel(data_arr, class_labels, 200, 0.0001, 100, ('lin', k1))
    data_mat = mat(data_arr)
    labels_mat = mat(class_labels).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_v_s = data_mat[sv_ind]
    label_sv = labels_mat[sv_ind]
    print('支持向量个数:%d' % shape(s_v_s)[0])
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_v_s, data_mat[i, :], ('lin', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(class_labels[i]): error_count += 1
    print('训练集错误率: %.2f%%' % ((float(error_count) / m) * 100))

    data_arr, class_labels = loadDataSet('machinelearninginaction/Ch06/testSetRBF2.txt')
    data_mat = mat(data_arr)
    labels_mat = mat(class_labels).transpose()
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_v_s, data_mat[i, :], ('lin', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(class_labels[i]): error_count += 1
    print('测试集错误率: %.2f%%' % ((float(error_count) / m) * 100))

    # data_arr, class_labels = loadDataSet('machinelearninginaction/Ch06/testSet.txt')
    # b, alphas = smoP_kernel(data_arr, class_labels, 200, 0.0001, 100, ('lin', k1))
    # data_mat = mat(data_arr)
    # labels_mat = mat(class_labels).transpose()
    # sv_ind = nonzero(alphas.A > 0)[0]
    # s_v_s = data_mat[sv_ind]
    # label_sv = labels_mat[sv_ind]
    # print('支持向量个数:%d' % shape(s_v_s)[0])
    # m, n = shape(data_mat)
    # error_count = 0
    # for i in range(m):
    #     kernel_eval = kernel_trans(s_v_s, data_mat[i, :], ('lin', k1))
    #     predict = kernel_eval.T * multiply(label_sv,alphas[sv_ind]) + b
    #     if sign(predict) != sign(class_labels[i]): error_count += 1
    # print('训练集错误率: %.2f%%' % ((float(error_count)/m)*100))
    # w = calc_ws(alphas, data_arr, class_labels)
    # show_classifier(data_arr, class_labels, w, b, alphas)


def img_vector(file_name):
    """
        Function:
            将32x32的二进制图像转换为1x1024向量
        Parameters:
            file_name - 文件名
        Returns:
            return_vect - 二进制图像的1x1024向量
        Modify:
            2018-09-26
    """
    return_vect = zeros((1, 1024))
    f = open(file_name, 'r')
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            return_vect[0,32 * i + j] = int(line_str[j])
    return return_vect


def load_image(dir_name):
    """
        Function:
            加载图片
        Parameters:
            dir_name - 图片文件夹路径
        Returns:
            training_mat - 数据矩阵
            hw_labels - 数据标签
        Modify:
            2018-09-26
    """
    from os import listdir
    hw_labels = []
    training_file_list = listdir(dir_name)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9: hw_labels.append(-1)
        else: hw_labels.append(1)
        training_mat[i,:] = img_vector('%s/%s' % (dir_name, file_name_str))
    return training_mat, hw_labels


def test_digits(k_tup=('rbf', 10)):
    """
        Function:
            基于SVM的手写数字识别测试函数
        Parameters:
            k_tup - 包含核函数信息的元组
        Returns:
            无
        Modify:
            2018-09-26
    """
    data_arr, label_arr = load_image('./machinelearninginaction/Ch02/digits/trainingDigits')
    b, alphas = smoP_kernel(data_arr, label_arr, 200, 0.0001, 10, k_tup)
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_v_s = data_mat[sv_ind]
    label__sv = label_mat[sv_ind]
    print('支持向量个数:%d' % np.shape(s_v_s)[0])
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_v_s, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label__sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print('训练集错误率: %.2f%%' % (float(error_count) / m))
    data_arr, label_arr = load_image('./machinelearninginaction/Ch02/digits/testDigits')
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_v_s, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label__sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print('测试集错误率: %.2f%%' % (float(error_count) / m))

if __name__ == '__main__':
    # data_arr, class_labels = loadDataSet('./machinelearninginaction/Ch06/testSetRBF.txt')
    # show_data_set(data_arr, class_labels)

    # test_rbf()

    test_digits(('rbf', 20))