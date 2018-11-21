# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.font_manager import FontProperties


def load_simp_data():
    """
    Function:
        加载数据集
    Parameters:
        无
    Returns:
        data_mat - 数据矩阵
        class_labels - 标签列表
    Modify:
        2018-10-16
    """
    data_mat = matrix([[1, 2.1],
                       [2, 1.1],
                       [1.3, 1],
                       [1, 1],
                       [2, 1]
                       ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def show_data_set(data_mat, lable_mat):
    """
        Function:
            数据集可视化
        Parameters:
            data_mat - 数据矩阵
            lable_mat - 标签列表
        Returns:
            无
        Modify:
            2018-10-16
        """
    data_plus = []
    data_minus = []
    for i in range(len(data_mat)):
        if lable_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    data_plus_np = array(data_plus)
    data_minus_np = array(data_minus)
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1])
    plt.show()


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """
    Function:
        通过阈值比较对数据进行分类
    Parameters:
        data_matrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        thresh_val - 阈值
        thresh_ineq - 标志
    Returns:
        ret_array - 分类结果
    Modify:
        2018-10-16
    """
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    """
    Function:
        建立单层决策树，找出最小误差率的划分阈值
    Parameters:
        data_arr - 数据矩阵
        class_labels - 数据标签
        D - 样本权重
    Returns:
        best_stump - 最佳单层决策树信息
        min_error - 最小误差
        best_class_est - 最佳的分类结果
    Modify:
        2018-10-16
    """
    data_matrix = mat(data_arr)
    # 标签转成列向量
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m, 1)))
    # 最小错误率初始化为+∞
    min_error = inf
    # 对每一维的特征，
    for i in range(n):
        # 找这一维特征的最大最小值
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        # 计算步长
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            # 大于阈值和小于阈值的情况，均遍历，lt:less than，gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                thresh_val = (range_min + float(j) * step_size)
                # 计算分类结果
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                # 初始化误差矩阵
                err_arr = mat(ones((m, 1)))
                # 将错误向量中分类正确项置0
                err_arr[predicted_vals == label_mat] = 0
                # 计算误差
                weighted_error = D.T * err_arr
                # print('split:dim %d, thresh %.2f,thresh inequal:%s, the weghted error is %.3f' % ( i, thresh_val, inequal, weighted_error))
                # 找到误差最小的分类方式
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it=40):
    """
    Function:
        基于单层决策树的Adboost算法
    Parameters:
        data_arr - 数据矩阵
        class_labels - 数据标签
        num_it - 迭代次数
    Returns:
        weak_class_arr - 单层决策树
    Modify:
        2018-10-17
    """
    weak_class_arr = []
    m = shape(data_arr)[0]
    # 初始化权重
    D = mat(ones((m, 1)) / m)
    agg_class_est = mat(zeros((m, 1)))
    for i in range(num_it):
        # 构建单层决策树
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        # print('D:', D.T)
        # 计算弱学习算法权重alpha,max(error, 1e-16)确保没有错误是发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        best_stump['alpha'] = alpha
        # 存储单层决策树
        weak_class_arr.append(best_stump)
        # print('class_est:', class_est.T)
        # 计算e的指数项
        expon = multiply(-1 * alpha * mat(class_labels).T, class_est)
        D = multiply(D, exp(expon))
        # 根据样本权重公式，更新样本权重
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        agg_class_est += alpha * class_est
        # print('aggClassEst', agg_class_est.T)
        # 计算误差
        aggErrors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate, '\n')
        # 误差为0，退出循环
        if errorRate == 0.0: break
    return weak_class_arr, agg_class_est


def ada_classify(data_to_class, classifier_arr):
    """
    Function:
        AdaBoost分类函数
    Parameters:
        data_to_class - 待分类样例
        classifier_arr - 训练好的分类器
    Returns:
        分类结果
    Modify:
        2018-10-17
    """
    data_matrix = mat(data_to_class)
    m = shape(data_matrix)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        # print(agg_class_est)
    # 如果agg_class_est大于0则返回+1，小于0则返回-1
    return sign(agg_class_est)


def load_data_set(file_name):
    """
    Function:
        自适应数据加载函数
    Parameters:
        file_name - 文件名
    Returns:
        data_mat - 数据列表
        label_mat - 标签列表
    Modify:
        2018-10-17
    """
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_roc(pred_strengths, class_labels):
    """
    Function:
        ROC曲线的绘制及AUC计算函数
    Parameters:
        pred_strengths - 分类器的预测强度
        class_labels - 类别
    Returns:
        无
    Modify:
        2018-10-19
    """
    font = FontProperties(fname=r'c:\\windows\fonts\simsun.ttc', size=4)
    cur = (1.0, 1.0)
    y_sum = 0.0
    # 统计正类的数量
    num_pos_clas = sum(array(class_labels) == 1.0)
    # y轴步长
    y_step = 1 / float(num_pos_clas)
    # x轴步长
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    # 预测强度排序
    sorted_indicies = pred_strengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            # 高度累加
            y_sum += cur[1]
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        # 更新绘制光标的位置
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    # 计算AUC
    print('AUC面积为：', y_sum * x_step)
    plt.show()


if __name__ == '__main__':
    # data_mat, class_labels = load_simp_data()
    # show_data_set(data_mat, class_labels)


    # data_mat, class_labels = load_simp_data()
    # D = mat(ones((5, 1)) / 5)
    # best_stump, min_error, best_class_est = build_stump(data_mat, class_labels, D)
    # print('best_stump:\n', best_stump)
    # print('min_error:\n', min_error)
    # print('best_class_est:\n', best_class_est)


    # data_mat, class_labels = load_simp_data()
    # weak_class_arr = adaboost_train_ds(data_mat, class_labels, 30)
    # print(ada_classify([[5, 5], [0, 0]], weak_class_arr))


    # train_data_arr, train_label_arr = load_data_set('./machinelearninginaction/Ch07/horseColicTraining2.txt')
    # classifier_array = adaboost_train_ds(train_data_arr, train_label_arr, 10)
    # print(classifier_array)
    # predictions = ada_classify(train_data_arr, classifier_array)
    # err_arr = mat(ones((len(train_data_arr), 1)))
    # print('训练集的错误率:%.3f%%' % float(err_arr[predictions != np.mat(train_label_arr).T].sum() / len(train_data_arr) * 100))
    # test_data_arr, test_label_arr = load_data_set('./machinelearninginaction/Ch07/horseColicTest2.txt')
    # predictions = ada_classify(test_data_arr, classifier_array)
    # err_arr = mat(ones((len(test_data_arr), 1)))
    # print('测试集的错误率:%.3f%%' % float(err_arr[predictions != np.mat(test_label_arr).T].sum() / len(test_data_arr) * 100))


    # train_data_set, train_labels = load_data_set('./machinelearninginaction/Ch07/horseColicTraining2.txt')
    # test_data_set, test_labels = load_data_set('./machinelearninginaction/Ch07/horseColicTest2.txt')
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME', n_estimators=10)
    # bdt.fit(train_data_set, train_labels)
    # predictions = bdt.predict(train_data_set)
    # err_arr = mat(ones((len(train_data_set), 1)))
    # print('训练集的错误率:%.3f%%' % float(err_arr[predictions != train_labels].sum() / len(train_data_set) * 100))
    # predictions = bdt.predict(test_data_set)
    # err_arr = mat(ones((len(test_data_set), 1)))
    # print('测试集的错误率:%.3f%%' % float(err_arr[predictions != test_labels].sum() / len(test_data_set) * 100))


    data_mat, class_labels = load_data_set('./machinelearninginaction/Ch07/horseColicTraining2.txt')
    weak_class_arr, agg_class_est = adaboost_train_ds(data_mat, class_labels, 30)
    plot_roc(agg_class_est.T, class_labels)
