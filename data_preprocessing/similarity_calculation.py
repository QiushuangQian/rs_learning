from numpy import *


# 欧氏距离：m维空间中两个点的真实距离
def EuclideanDistance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


print("a，b二维欧氏距离为：", EuclideanDistance((1, 1), (2, 2)))


# 曼哈顿距离：城市街区距离
def ManhattanDistance(a, b):
    return abs(a[0] - a[1]) + abs(b[0] - b[1])


print("a，b二维曼哈顿距离为：", ManhattanDistance((1, 1), (2, 2)))


# 切比雪夫距离
def ChebyshevDistance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


print("a，b二维切比雪夫距离为：", ChebyshevDistance((1, 2), (3, 4)))

# 马氏距离：数据的协方差距离
pass


# 夹角余弦距离：几何中用来衡量两个向量方向的差异，机器学习中用于衡量样本向量之间的差异
def CosineSimilarity(a, b):
    cos = (a[0] * b[0] + a[1] * b[1]) / (sqrt(a[0] ** 2 + a[1] ** 2) * sqrt(b[0] ** 2 + b[1] ** 2))
    return cos


print("a，B二维夹角余弦距离为：", CosineSimilarity((1, 1), (2, 2)))


# 杰卡德相似系数：A、B两个集合的交集元素在其并集中所占的比例称为杰卡德相似系数

def JacccardSimilarityCoefficient(a, b):
    set_a = set(a)
    set_b = set(b)
    dis = float(len((set_a & set_b)) / len(set_a | set_b))
    return dis


print("a，b杰卡德相似系数为：", JacccardSimilarityCoefficient((1, 2, 3), (2, 3, 4)))


# 杰卡德距离：两个集合中不同元素站所有元素的比例，用于衡量两个集合的区分度
def JacccardSimilarityDistance(a, b):
    set_a = set(a)
    set_b = set(b)
    dis = float(len((set_a | set_b) - (set_a & set_b)) / len(set_a | set_b))
    return dis


print("a，b杰卡德距离为：", JacccardSimilarityDistance((1, 2, 3), (2, 3, 4)))
