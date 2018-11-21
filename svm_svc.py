import numpy as np
import time
from os import listdir
from sklearn.svm import SVC


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
    return_vect = np.zeros((1, 1024))
    f = open(file_name, 'r')
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def hand_writing_class_test():
    """
        Function:
            手写数字分类测试
        Parameters:
            无
        Returns:
            无
        Modify:
            2018-09-26
    """
    hw_labels = []
    training_file_list = listdir('./machinelearninginaction/Ch02/digits/trainingDigits')
    print(training_file_list)
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        class_num_str = int(file_name_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img_vector('./machinelearninginaction/Ch02/digits/trainingDigits/%s' % (file_name_str))
    clf = SVC(C=200, kernel='rbf')
    clf.fit(training_mat, hw_labels)
    test_file_list = listdir('./machinelearninginaction/Ch02/digits/testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        class_num_str = int(file_name_str.split('_')[0])
        vector_under_test = img_vector('./machinelearninginaction/Ch02/digits/testDigits/%s' % (file_name_str))
        classifier_result = clf.predict(vector_under_test)
        print('分类返回结果为%d\t真实结果为%d' % (classifier_result, class_num_str))
        if (classifier_result != class_num_str):
            error_count += 1.0
    print('总共错了%d个数据\n错误率为%f%%' % (error_count, error_count / m_test * 100))


if __name__ == '__main__':
    start_time = time.time()
    hand_writing_class_test()
    end_time = time.time()
    print('程序总共执行时长：%.1fs' % (end_time - start_time))
