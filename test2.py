# -*- conding: utf-8 -*-
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    plot_data_set()