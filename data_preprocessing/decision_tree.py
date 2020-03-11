'''
决策树：
        回归决策树：对连续变量构建决策树
        分类决策树：对离散变量构建决策树
构建过程：
1 树从代表训练样本的根结点开始
2 若样本都在同一类中，则该节点为树叶，并用该类标记
3 否则，算法选择最有分类能力的属性作为决策树的当前节点
4 根据当前决策节点属性取值的不同，将训练样本数据集data分为若干子集，每个取值形成一个分支
5 针对4得到的每一个子集，重复1、2、3步骤，递归形成每个划分样本上的决策树。一旦一个属性只出现在一个节点上，就不必在该节点的任何子节点考虑它

递归结束标志：
1 给定节点的所有样本属于同一类
2 没有剩余属性可以用来进一步划分样本
3 若某一分支没有满足该分支中已有分类的样本，则以样本的多数类创建一个树叶
'''

'''
数据对应关系：
天气：晴（2）阴（1）雨（0）
温度：炎热（2）适中（1）寒冷（0）
湿度：高（1）低（0）
风速：强（1）弱（0）
举办活动：是（yes）否（no）
'''

import operator
import math


class DecisionTree:
    def __init__(self):
        pass

    # 加载数据
    def loadData(self):
        data = [
            [2, 2, 1, 0, "yes"],
            [2, 2, 1, 1, "no"],
            [1, 2, 1, 0, "yes"],
            [0, 0, 0, 0, "yes"],
            [0, 0, 0, 1, "no"],
            [1, 0, 0, 1, "yes"],
            [2, 1, 1, 0, "no"],
            [2, 0, 0, 0, "yes"],
            [0, 1, 0, 0, "yes"],
            [2, 1, 0, 1, "yes"],
            [1, 2, 0, 0, "no"],
            [0, 1, 1, 1, "no"],
        ]

        # 分类属性
        features = ["天气", "温度", "湿度", "风速"]
        return data, features

    # 计算给定数据集的香农熵
    def ShannonEnt(self, data):
        # 求长度
        numData = len(data)
        labelCounts = {}
        for feature in data:
            # 获得标签
            oneLabel = feature[-1]
            # 若标签不在新定义的字典里，则创建该标签
            labelCounts.setdefault(oneLabel, 0)
            # 该类标签下含有数据的个数
            labelCounts[oneLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            # 同类标签出现的概率
            prob = float(labelCounts[key]) / numData
            # 以2为底的对数
            shannonEnt -= prob * math.log2(prob)
        return shannonEnt

    # 选择最好的划分属性标签：针对原数据集，遍历所有属性，计算根据每个属性划分数据对应的香农熵Ei，然后计算每个属性的信息增益Gi
    # 划分数据集，三个参数为带划分的数据集，划分数据集的特征，特征的返回值
    def splitData(self, data, axis, value):
        retData = []
        for feature in data:
            if feature[axis] == value:
                # 抽取有相同特征的数据集
                reducedFeature = feature[:axis]
                reducedFeature.extend(feature[axis + 1:])
                retData.append(reducedFeature)
        return retData

    def chooseBestFeatureToSplit(self, data):
        numFeature = len(data[0]) - 1
        baseEntropy = self.ShannonEnt(data)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeature):
            # 获取第i个特征所有的可能取值
            featureList = [result[i] for result in data]
            # 从列表中创建集合，得到不重复的所有可能取值
            uniqueFeatureList = set(featureList)
            newEntropy = 0.0
            for value in uniqueFeatureList:
                # 以i为数据集特征，value为返回值，划分数据集
                splitDataSet = self.splitData(data, i, value)
                # 数据集特征为i的数据集所占比例
                prob = len(splitDataSet) / float(len(data))
                # 计算每种数据集的信息熵
                newEntropy += prob * self.ShannonEnt(splitDataSet)
            infoGain = baseEntropy - newEntropy
            # 计算机最好的信息增益，增益越大说明所占决策权越大
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 递归构建决策树
    def majorityCnt(self, labelsList):
        labelsCount = {}
        for vote in labelsList:
            if vote not in labelsCount.keys():
                labelsCount[vote] = 0
            labelsCount[vote] += 1
        # 排序，True升序
        sortedLabelsCount = sorted(
            labelsCount.iteritems(), key=operator.itemgetter(1), reverse=True
        )
        # 返回出现次数最多的
        print(sortedLabelsCount)
        return sortedLabelsCount[0][0]

    # 创建决策树
    def createTree(self, data, features):
        # 使用"="产生的新变量
        features = list(features)
        labelsList = [line[-1] for line in data]
        # 类别完全相同则停止划分
        if labelsList.count(labelsList[0]) == len(labelsList):
            return labelsList[0]
        # 遍历完所有特征值时返回出现次数最多的
        if len(data[0]) == 1:
            return self.majorityCnt(labelsList)
        # 选择最好的数据集划分方式
        bestFeature = self.chooseBestFeatureToSplit(data)
        # 得到对应的标签值
        bestFeatLabel = features[bestFeature]
        myTree = {bestFeatLabel: {}}
        # 清空features[bestFeat],在下一次使用时清零
        del (features[bestFeature])
        featureValues = [example[bestFeature] for example in data]
        uniqueFeatureValues = set(featureValues)
        for value in uniqueFeatureValues:
            subFeatures = features[:]
            # 递归调用创建决策树函数
            myTree[bestFeatLabel][value] = self.createTree(
                self.splitData(data, bestFeature, value), subFeatures
            )
        return myTree

    # 预测新数据特征下是否举办活动
    def predict(self, tree, features, x):
        for key1 in tree.keys():
            secondDict = tree[key1]
            # key是根节点代表的特征，featIndex是取根节点特征在特征列表中的索引，方便后面对输入样本逐变量判断
            featIndex = features.index(key1)
            # 这里每个key值对应的是根节点特征的不同取值
            for key2 in secondDict.keys():
                # 找到输入样本在决策树中由根节点往下走的路径
                if x[featIndex] == key2:
                    # 该分支产生了一个内部节点，则在决策树中继续用同样的操作查找路径
                    if type(secondDict[key2]).__name__ == "dict":
                        classLabel = self.predict(secondDict[key2], features, x)
                    # 该分支产生的是叶节点，直接取值就得到类别
                    else:
                        classLabel = secondDict[key2]
        return classLabel


if __name__ == "__main__":
    dtree = DecisionTree()
    data, features = dtree.loadData()
    myTree = dtree.createTree(data, features)
    print(myTree)
    label = dtree.predict(myTree, features, [1, 1, 1, 0])
    print("新数据[1,1,1,0]对应的是否要举办活动为：{}".format(label))
