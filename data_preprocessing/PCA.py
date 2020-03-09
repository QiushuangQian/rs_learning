'''
PCA主成分分析
通过正交变换，将一组可能存在相关性的变量转换为一组线性不相关的变量，转换后的这组变量叫主成分
核心知识点：协方差矩阵和特征值分解
降维过程：
1 对特征值进行标准化
2 计算协方差矩阵
3 计算协方差矩阵的特征值和特征向量
4 选取最大的k个特征值对应的特征向量，得到特征向量矩阵
5 将数据变换为k维，得到新的数据集
'''

import numpy as np
from sklearn import datasets


# 鸢尾花数据特征集进行降维
class PCATest:
    def __init__(self):
        pass

    # 加载鸢尾花数据集中的特征作为PCA的原始数据集并进行标准化
    def loadIris(self):
        data = datasets.load_iris()["data"]
        return data

    # 标准化数据
    def Standard(self, data):
        # axis=0 按列取均值
        mean_vector = np.mean(data, axis=0)
        return mean_vector,data-mean_vector

    # 计算协方差矩阵
    def getCovMatrix(self, newData):
        # rowvar = 0 表示数据的每一列代表一个feature特征
        return np.cov(newData, rowvar=0)

    # 计算协方差矩阵的特征值和特征向量
    def getFValueAndFVector(self, covMatrix):
        fValue, fVector = np.linalg.eig(covMatrix)
        return fValue, fVector

    # 得到特征向量矩阵
    def getVectorMatrix(self, fValue, fVector, k):
        fValueSort = np.argsort(fValue)
        fVAlueTopN = fValueSort[:-(k + 1):-1]
        return fVector[:, fVAlueTopN]

    # 得到降维后的数据
    def getResult(self, data, vectorMatrix):
        return np.dot(data, vectorMatrix)


if __name__ == '__main__':
    # 创建PCA对象
    pactest = PCATest()
    # 加载Iris数据集
    data = pactest.loadIris()
    # 归一化数据
    mean_vector, newData = pactest.Standard(data)
    # 得到协方差矩阵
    covMatrix = pactest.getCovMatrix(newData)
    print("协方差矩阵为：\n{}".format(covMatrix))
    # 得到特征值和特征向量
    fValue, fVector = pactest.getFValueAndFVector(covMatrix)
    print("特征值为：\n{}".format(fValue))
    print("特征向量为：\n{}".format(fVector))
    # 得到要降到k维的特征向量矩阵
    vectorMatrix = pactest.getVectorMatrix(fValue, fVector, k=2)
    print("k维特征向量矩阵为：\n{}".format(vectorMatrix))
    # 计算结果
    result = pactest.getResult(newData, vectorMatrix)
    print("最终降维结果为：\n{}".format(result))

    # 得到重构数据
    print("最终重构数据为：\n{}".format(np.mat(result) * vectorMatrix.T + mean_vector))
