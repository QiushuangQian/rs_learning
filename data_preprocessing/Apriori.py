'''
关联分析：找到具有某种相关性的物品
Apriori：算法使用频繁项集的所有非空子集也一定是频繁的
相关概念：
项集（ItemSet）：包含0个或多个项的集合
频繁项集：经常一起出现的物品集合
支持度计数（SupportCount）：一个项集出现的次数
项集支持度：一个项集出现的次数与数据集所有事物的百分比
项集置信度：数据集中同时包含A、B的百分比
Apriori算法过程分析：
1 通过扫描数据库，累计每个项的计数，并收集满足最小支持度的项，找出频繁1项集的集合，该集合记为L1
2 使用L1找出频繁2项集的集合L2，使用L2找出L3
3 如此下去，直至不能再找到频繁k项集，每找出一个Lk需要一次完整的数据库扫描
'''


class Apriori:
    def __init__(self, minSupport, minConfidence):
        # 最小支持度
        self.minSupport = minSupport
        # 最小置信度
        self.minConfidence = minConfidence
        self.data = self.loadData()

    # 加载数据集
    def loadData(self):
        return [[1, 5], [2, 3, 4], [2, 3, 4, 5], [2, 3]]

    # 生成项集C1，不包含项集中每个元素出现的次数
    def createC1(self, data):
        # C1为大小为1的项的集合
        C1 = list()
        for items in data:
            for item in items:
                if [item] not in C1:
                    C1.append([item])
        # map函数表示遍历C1中的每一个元素执行forzenset
        # frozenset表示冰冻的集合，即不可改变
        return list(map(frozenset, sorted(C1)))

    # 生成频繁项集
    '''
    步骤：
    1 扫描初始候选项集C1，生成项集L1和所有的项集组合SupporData
    2 将项集L1加入L中进行记录（L为每次迭代后符合支持度的项集）
    3 根据L1生成新的候选项集C2
    4 扫描候选项集C2，生成项集L2和所有的项集集合
    5 更新项集集合SuppoData和L
    6 重复步骤2-5，直到项集中的元素为全部元素时停止迭代
    '''

    # 此函数用于从候选项集Ck生成Lk
    def scanD(self, Ck):
        # Data 表示数据列表的列表
        Data = list(map(set, self.data))
        CkCount = {}
        # 统计Ck项集中每个元素出现的次数
        for items in Data:
            for one in Ck:
                # issubset:表示如果集合one中的每一个元素都在items中则返回true
                if one.issubset(items):
                    CkCount.setdefault(one, 0)
                    CkCount[one] += 1
        # 数据条数
        numItems = len(list(Data))
        # 初始化符合支持度的项集
        Lk = []
        # 初始化所有符合条件的项集以及对应的支持度
        supportData = {}
        for key in CkCount:
            # 计算每个项集的支持度，若满足条件则把该项集加入到Lk中
            support = CkCount[key] * 1.0 / numItems
            if support >= self.minSupport:
                Lk.insert(0, key)
            # 构建支持的项集字典
            supportData[key] = support
        return Lk, supportData

    # generateNewCk的输入参数为频繁项集列表Lk与项集元素个数k，输出为Ck
    def generateNewCk(self, Lk, k):
        nextLk = []
        lenLk = len(Lk)
        # 若两个项集的长度为k-1，则必须前k-2项相同才可连接，即求并集，所以[:k-2]的实际作用为取列表的前k-1个元素
        for i in range(lenLk):
            for j in range(i + 1, lenLk):
                # 前k-2项相同时合并两个集合
                L1 = list(Lk[i])[:k - 2]
                L2 = list(Lk[j])[:k - 2]
                if sorted(L1) == sorted(L2):
                    nextLk.append(Lk[i] | Lk[j])
        return nextLk

    # 生成频繁项集
    def generateLk(self):
        # 构建候选项集C1
        C1 = self.createC1(self.data)
        L1, supportData = self.scanD(C1)
        L = [L1]
        k = 2
        while len(L[k - 2]) > 0:
            # 组合项集中的元素，生成新的候选项集Ck
            Ck = self.generateNewCk(L[k - 2], k)
            Lk, supK = self.scanD(Ck)
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    # 生成相关规则
    '''
    生成规则：
    1 对于每个频繁项集itemset，生成itemset的所有非空子集
    2 对于itemset的每个非空子集s，若s的置信度大于设置的最小置信度，则输出对应的相关规则
    '''

    def generateRules(self, L, supportData):
        # 最终记录的相关规则结果
        ruleResult = []
        for i in range(1, len(L)):
            for ck in L[i]:
                Cks = [frozenset([item]) for item in ck]
                # 频繁项集中有三个或三个以上元素的组合
                self.rulesofMore(ck, Cks, supportData, ruleResult)
        return ruleResult

    # 频繁项集只有两个元素
    def rulesofTwo(self, ck, Cks, supportData, ruleResult):
        prunedH = []
        for oneCk in Cks:
            # 计算置信度
            conf = supportData[ck] / supportData[ck - oneCk]
            if conf > self.minConfidence:
                print(ck - oneCk, "-->", oneCk, "Confidence is:", conf)
                ruleResult.append((ck - oneCk, oneCk, conf))
                prunedH.append(oneCk)
        return prunedH

    # 频繁项集有三个及三个以上的元素，递归生成相关规则
    def rulesofMore(self, ck, Cks, supportData, ruleResult):
        m = len(Cks[0])
        while len(ck) > m:
            Cks = self.rulesofTwo(ck, Cks, supportData, ruleResult)
            if len(Cks) > 1:
                Cks = self.generateNewCk(Cks, m + 1)
                m += 1
            else:
                break


if __name__ == "__main__":
    apriori = Apriori(minSupport=0.5, minConfidence=0.6)
    L, supportData = apriori.generateLk()
    for one in L:
        print("项数为%s的频繁项集为：" % (L.index(one) + 1), one)
    print("supportData:", supportData)
    print("minConfidence为0.6时：")
    rules = apriori.generateRules(L, supportData)
