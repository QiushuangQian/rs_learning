"""
K最近邻算法
KNN算法核心思路：
1 计算未知样本和每个训练样本的距离distane
2 按照距离distance的递增关系排序
3 得到距离最小的前K个样本（K应为奇数）
4 统计K最近邻样本中每个类标号出现的次数
5 选择出现频率最高的类标号作为未知样本的类标号
"""

import numpy as np


class KNN:
    def __init__(self, k):
        self.K = k

    # 准备数据
    def createData(self):
        features = np.array([[180, 76], [158, 43], [176, 78], [161, 49]])
        labels = ["男", "女", "男", "女"]
        return features, labels

    # 数据标准化（Min-Max）
    def Normalization(self, data):
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        new_data = (data - mins) / (maxs - mins)
        return new_data, maxs, mins

    # 计算K最近邻
    def classfy(self, one, data, labels):
        # 计算新样本与数据集中每个样本之间的距离（欧氏距离实现）
        differenceData = data - one
        squareData = (differenceData ** 2).sum(axis=1)
        distance = squareData ** 0.5
        sortDistanceIndex = distance.argsort()
        # 统计K最近邻的label
        labelCount = dict()
        for i in range(self.K):
            label = labels[sortDistanceIndex[i]]
            labelCount.setdefault(label, 0)
            labelCount[label] += 1

        #计算结果
        sortLabelCount = sorted(labelCount.items(), key= lambda x:x[1],reverse=True)
        print(sortLabelCount)
        return sortLabelCount[0][0]


if __name__ == '__main__':
    knn = KNN(3)
    features,labels = knn.createData()
    new_data,maxs,mins = knn.Normalization(features)
    #新数据标准化
    one = np.array([176,76])
    new_one = (one-mins)/(maxs-mins)
    #求得结果
    result = knn.classfy(new_one,new_data,labels)
    print("数据{}的预测性别为：{}".format(one,result))
