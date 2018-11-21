# -*- conding: utf-8 -*-
import matplotlib.pyplot as plt
from numpy import *
from matplotlib.font_manager import FontProperties


def load_data_set(file_name):
    """
    Function:
        加载数据集
    Parameters:
        file_name - 文件名
    Returns:
        data_mat - 数据列表
        label_mat - 标签列表
    Modify:
        2018-10-22
    """
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_data_set():
    """
    Function:
        绘制数据集
    Parameters:
        无
    Returns:

    Modify:
        2018-10-22
    """
    data_mat, label_mat = load_data_set('./machinelearninginaction/Ch08/ex0.txt')
    n = len(data_mat)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(data_mat[i][1])
        ycord.append(label_mat[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('data_set')
    plt.xlabel('X')
    plt.show()


def stand_regres(x_arr, y_arr):
    """
    Function:
        计算回归系数w
    Parameters:
        x_arr - 数据列表
        y_arr - 标签列表
    Returns:
        ws - 回归系数
    Modify:
        2018-10-22
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def plot_regression():
    """
    Function:
        加载数据集
    Parameters:
        无
    Returns:
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-10-22
    """
    data_mat, label_mat = load_data_set('./machinelearninginaction/Ch08/ex0.txt')
    ws = stand_regres(data_mat, label_mat)
    x_mat = mat(data_mat)
    y_mat = mat(label_mat)
    x_coyp = x_mat.copy()
    x_coyp.sort(0)
    y_hat = x_coyp * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_coyp[:, 1], y_hat, c='red')
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    plt.title('data_set')
    plt.xlabel('X')
    plt.show()


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """
    Function:
        使用局部加权线性回归计算回归系数w
    Parameters:
        test_point -  测试样本点
        x_arr - x数据集
        y_arr - y数据集
    Returns:
        ws - 回归系数
    Modify:
        2018-10-24
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    # 生成对角矩阵
    weights = mat(eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    x_t_x = x_mat.T * (weights * x_mat)
    if linalg.det(x_t_x) == 0.0:
        print('矩阵为奇异矩阵,不能求逆')
        return
    # 计算回归系数
    ws = x_t_x.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    """
    Function:
        局部加权线性回归测试
    Parameters:
        test_arr -  测试数据集
        x_arr - x数据集
        y_arr - y数据集
    Returns:
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-10-24
    """
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def plot_lwlr_regression():
    """
    Function:
        绘制多条局部加权回归曲线
    Parameters:
        无
    Returns:
        无
    Modify:
        2018-10-24
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    x_arr, y_arr = load_data_set('./machinelearninginaction/Ch08/ex0.txt')
    y_hat_1 = lwlr_test(x_arr, x_arr, y_arr, 1.0)
    y_hat_2 = lwlr_test(x_arr, x_arr, y_arr, 0.01)
    y_hat_3 = lwlr_test(x_arr, x_arr, y_arr, 0.003)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    # 排序，返回索引值
    srt_ind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srt_ind][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    # 绘制回归曲线
    axs[0].plot(x_sort[:, 1], y_hat_1[srt_ind], c='red')
    axs[1].plot(x_sort[:, 1], y_hat_2[srt_ind], c='red')
    axs[2].plot(x_sort[:, 1], y_hat_3[srt_ind], c='red')
    # 绘制原数据点
    axs[0].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[1].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[2].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    # plt.setp():设置图标实例的属性
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


def rss_error(y_arr, y_hat_arr):
    """
    Function:
        误差大小评价函数
    Parameters:
        y_arr - 真实数据
        y_hat_arr - 预测数据
    Returns:
        误差大小
    Modify:
        2018-10-24
    """
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regress(x_mat, y_mat, lam=0.2):
    """
    Function:
        岭回归
    Parameters:
        x_mat - x数据集
        y_mat - y数据集
        lam - 缩减系数
    Returns:
         ws - 回归系数
    Modify:
        2018-11-07
    """
    x_t_x = x_mat.T * x_mat
    denom = x_t_x + eye(shape(x_mat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('矩阵为奇异矩阵,不能转置')
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    """
    Function:
        岭回归测试
    Parameters:
        x_mat - x数据集
        y_mat - y数据集
        lam - 缩减系数
    Returns:
         w_mat - 回归系数矩阵
    Modify:
        2018-11-07
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    # 行与行操作，求均值
    y_mean = mean(y_mat, axis=0)
    # 数据减去均值
    y_mat = y_mat - y_mean
    # 行与行操作，求均值
    x_means = mean(x_mat, axis=0)
    # 行与行操作，求方差
    x_var = var(x_mat, axis=0)
    # 数据减去均值除以方差实现标准化
    x_mat = (x_mat - x_means) / x_var
    # 30个不同的lambda测试
    num_test_pts = 30
    # 初始回归系数矩阵
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        # lambda以e的指数变化，最初是一个非常小的数
        ws = ridge_regress(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def plot_ridge_regress_mat():
    """
    Function:
        绘制岭回归系数矩阵
    Parameters:
        无
    Returns:
        无
    Modify:
        2018-10-24
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    ab_x, ab_y = load_data_set('./machinelearninginaction/Ch08/abalone.txt')
    redge_weights = ridge_test(ab_x, ab_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redge_weights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def regularize(x_mat, y_mat):
    """
    Function:
        数据标准化
    Parameters:
        x_mat - x数据集
        y_mat - y数据集
    Returns:
        in_x_mat - 标准化后的x数据集
        in_y_mat - 标准化后的y数据集
    Modify:
        2018-11-11
    """
    in_x_mat = x_mat.copy()
    in_y_mat = y_mat.copy()
    in_y_mean = mean(y_mat, 0)
    in_y_mat = y_mat - in_y_mean
    in_means = mean(in_x_mat, 0)
    in_var = var(in_x_mat, 0)
    in_x_mat = (in_x_mat - in_means) / in_var
    return in_x_mat, in_y_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    """
    Function:
        绘制岭回归系数矩阵
    Parameters:
        无
    Returns:
        无
    Modify:
        2018-10-24
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    x_mat, y_mat = regularize(x_mat, y_mat)
    m, n = shape(x_mat)
    # 初始化num_it次迭代的回归系数矩阵
    return_mat = zeros((num_it, 1))
    # 初始化回归系数矩阵
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    # 迭代num_it次
    for i in range(num_it):
        # 打印当前回归系数矩阵
        print(ws.T)
        lowest_error = float('inf')
        # 遍历每个特征的回归系数
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                # 微调回归系数
                ws_test[j] += eps* sign
                # 计算预测值
                y_test = x_mat * ws_test
                # 计算平方误差
                rss_e = x_mat * ws_test
                # 如果误差更小，则更新当前的最佳回归系数
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        # 记录numIt次迭代的回归系数矩阵
        return_mat[i, :] = ws.T
    return return_mat




if __name__ == '__main__':
    # plot_data_set()

    # plot_regression()

    # data_mat, label_mat = load_data_set('./machinelearninginaction/Ch08/ex0.txt')
    # ws = stand_regres(data_mat, label_mat)
    # x_mat = mat(data_mat)
    # y_mat = mat(label_mat)
    # y_hat = x_mat * ws
    # print(corrcoef(y_hat.T, y_mat))

    # plot_lwlr_regression()

    # ab_x, ab_y = load_data_set('./machinelearninginaction/Ch08/abalone.txt')
    # print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    # y_hat_01 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
    # y_hat_1 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
    # y_hat_10 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
    # print('k=0.1时,误差大小为:', rss_error(ab_y[0:99], y_hat_01.T))
    # print('k=1  时,误差大小为:', rss_error(ab_y[0:99], y_hat_1.T))
    # print('k=10 时,误差大小为:', rss_error(ab_y[0:99], y_hat_10.T))
    # print('')
    # print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    # y_hat_1 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1)
    # y_hat_2 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1)
    # y_hat_3 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10)
    # print('k=0.1时,误差大小为:', rss_error(ab_y[100:199], y_hat_1.T))
    # print('k=1  时,误差大小为:', rss_error(ab_y[100:199], y_hat_2.T))
    # print('k=10 时,误差大小为:', rss_error(ab_y[100:199], y_hat_3.T))
    # print('')
    # print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    # print('k=1时,误差大小为:', rss_error(ab_y[100:199], y_hat_2.T))
    # ws = stand_regres(ab_x[0:99], ab_y[0:99])
    # y_hat = mat(ab_x[100:199]) * ws
    # print('简单的线性回归误差大小:', rss_error(ab_y[100:199], y_hat.T.A))

    plot_ridge_regress_mat()

