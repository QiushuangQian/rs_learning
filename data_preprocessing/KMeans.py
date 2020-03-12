'''
数据聚类：
KMeans算法基本原理：
1 随机初始化K个初始簇类中心，对应K个初始簇类，按照"距离最近"原则，将每条数据划分到最近的簇类
2 第一次迭代后，更新各个簇类中心，然后进行第二次迭代，依旧按照"距离最近"原则进行数据归类
3 知道簇类中心不再改变，或者前后变化小于给定的误差值，或者达到迭代次数，停止迭代

具体执行步骤：
1 在数据集中初始化K个簇类中心，对应K个初始簇类
2 计算给定数据集中每条数据到K个簇类中心的距离
3 按照"距离最近"原则，将每条数据都划分到最近的簇类中
4 更新每个簇类中心
5 迭代执行步骤2、3、4，直至簇类中心不再改变，或者前后变化小于给定的误差值，或者达到迭代次数
6 结束算法，输出最后的簇类中心和对应的簇类
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


if __name__ == "__main__":
    file = "../data/sku-price/skuid_price.csv"
    km = kMeans()
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    Cluster = km.kMeans(newData["price"].values, K=7, maxIters=200)
    print(Cluster)
