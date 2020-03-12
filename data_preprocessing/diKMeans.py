'''
二分-kMeans算法是分层聚类的一种
实现具体思路：
1 初始化簇类表，使之包含所有的数据
2 对每一个簇类应用k均值聚类算法（k=2）
3 计算划分后的误差，选择所有被划分的聚类中总误差最小的并保存
4 迭代步骤2、3，簇类数目到达k后停止

二分-kMeans优点：
加速了kMeans的执行速度，减少了相似度的计算次数
能够克服"kMeans收敛于局部最优"的缺点
'''

import numpy as np
import pandas as pd
import random


class kMeans:
    def __init__(self):
        pass

    # 加载数据集
    def loadData(self, file):
        return pd.read_csv(file, header=0, sep=",")

    # 去除异常值,使用正态分布的方法，同时保证最大异常值为5000，最小异常值为1
    def filterAnomalyValue(self, data):
        # 上限区间
        upper = np.mean(data["price"]) + 3 * np.std(data["price"])
        # 下限区间
        lower = np.mean(data["price"]) - 3 * np.std(data["price"])
        upper_limit = upper if upper > 5000 else 5000
        lower_limit = lower if lower < 1 else 1
        print("最大异常值为：{}，最小异常值为：{}".format(upper_limit, lower_limit))
        # 过滤掉大于最大异常值和最小异常值的值
        newData = data[(data["price"] < upper_limit) & data["price"] > lower_limit]
        return newData, upper_limit, lower_limit

    # 初始化聚类中心
    def initCenters(self, values, K, Cluster):
        random.seed(100)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0, len(values))
            Cluster.setdefault(i, {})
            Cluster[i]["center"] = values[index]
            Cluster[i]["values"] = []

            oldCenters.append(values[index])
        return oldCenters, Cluster

    # 进行聚类
    # 计算任意两条数据之间的欧氏距离
    def distance(self, price1, price2):
        return np.emath.sqrt(pow(price1 - price2, 2))

    # 聚类
    def kMeans(self, data, K, maxIters):
        # 最终聚类结果
        Cluster = dict()
        oldCenters, Cluster = self.initCenters(data, K, Cluster)
        print("初始的簇类中心为：{}".format(oldCenters))
        # 标志变量，若为True，则继续迭代
        clusterChanged = True
        # 记录迭代次数 最大迭代
        i = 0
        while clusterChanged:
            for price in data:
                # 每条数据与最近簇类中心的距离，初始化为正无穷大
                minDistance = np.inf
                # 每条数据对应的索引，初始化为-1
                minIndex = -1
                for key in Cluster.keys():
                    # 计算每条数据到簇类中心的距离
                    dis = self.distance(price, Cluster[key]["center"])
                    if dis < minDistance:
                        minDistance = dis
                        minIndex = key
                Cluster[minIndex]["values"].append(price)

            newCenters = list()
            for key in Cluster.keys():
                newCenter = np.mean(Cluster[key]["values"])
                Cluster[key]["center"] = newCenter
                newCenters.append(newCenter)
            print("第{}次迭代后的簇类中心为：{}".format(i, newCenters))
            if oldCenters == newCenters or i > maxIters:
                clusterChanged = False
            else:
                oldCenters = newCenters
                i += 1
                # 删除Cluster中记录的簇类值
                for key in Cluster.keys():
                    Cluster[key]["values"] = []
        return Cluster

    # 计算对应的SSE值（误差平方和）
    def SSE(self, data, mean):
        newData = np.mat(data) - mean
        return (newData * newData.T).tolist()[0][0]

    # 二分-kMeans算法
    def diKMeans(self, data, K=7):
        # 簇类对应的SSE值
        clusterSSEResult = dict()
        clusterSSEResult.setdefault(0, {})
        clusterSSEResult[0]["values"] = data
        clusterSSEResult[0]["sse"] = np.inf
        clusterSSEResult[0]["center"] = np.mean(data)

        while len(clusterSSEResult) < K:
            maxSSE = -np.inf
            maxSSEKey = 0
            # 找到最大SSE对应数据，进行kmeans聚类
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]["sse"] > maxSSE:
                    maxSSE = clusterSSEResult[key]["sse"]
                    maxSSEKey = key
            # clusterSSEResult{0:{'center':x,'values':[]},1:{'center':x,'values':[]}}
            clusterSSEResult = \
                self.kMeans(clusterSSEResult[maxSSEKey]["values"], K=2, maxIters=200)

            # 删除clusterSSE中的minKey对应的值
            del clusterSSEResult[maxSSEKey]
            # 将经过kMeans聚类后的结果赋值给clusterSSEResult
            clusterSSEResult.setdefault(maxSSEKey, {})
            clusterSSEResult[maxSSEKey]["center"] = clusterSSEResult[0]["center"]
            clusterSSEResult[maxSSEKey]["values"] = clusterSSEResult[0]["values"]
            clusterSSEResult[maxSSEKey]["sse"] = \
                self.SSE(clusterSSEResult[0]["values"], clusterSSEResult[0]["center"])

            maxKey = max(clusterSSEResult.keys()) + 1
            clusterSSEResult.setdefault(maxKey, {})
            clusterSSEResult[maxKey]["center"] = clusterSSEResult[1]["center"]
            clusterSSEResult[maxSSEKey]["values"] = clusterSSEResult[1]["values"]
            clusterSSEResult[maxSSEKey]["sse"] = \
                self.SSE(clusterSSEResult[1]["values"], clusterSSEResult[1]["center"])

        return clusterSSEResult


if __name__ == "__main__":
    file = "../data/sku-price/skuid_price.csv"
    km = kMeans()
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    # Cluster = km.kMeans(newData["price"].values, K=7, maxIters=200)
    clusterSSE = km.diKMeans(newData["price"].values, K=7)
    print(clusterSSE)
