import math


class UserCF:
    def __init__(self):
        self.user_score_dict = self.initUserScore()
        self.users_sim = self.userSimilarityBetter()

    # 初始化用户评分数据
    def initUserScore(self):
        user_score_dict = {
            'A': {'a': 3.0, 'b': 4.0, 'c': 0.0, 'd': 3.5, 'e': 0.0},
            'B': {'a': 4.0, 'b': 0.0, 'c': 4.5, 'd': 0.0, 'e': 3.5},
            'C': {'a': 0.0, 'b': 3.5, 'c': 0.0, 'd': 0.0, 'e': 3.0},
            'D': {'a': 0.0, 'b': 4.0, 'c': 0.0, 'd': 3.5, 'e': 3.0}
        }
        return user_score_dict

    # 计算用户之间的相似度，采用的是遍历每个用户进行计算
    def userSimilarity(self):
        W = dict()
        for u in self.user_score_dict.keys():
            W.setdefault(u, {})
            for v in self.user_score_dict.keys():
                if u == v:
                    continue
                u_set = set([key for key in self.user_score_dict[u].keys() if self.user_score_dict[u][key] > 0])
                v_set = set([key for key in self.user_score_dict[v].keys() if self.user_score_dict[v][key] > 0])
                W[u][v] = float(len(u_set & v_set)) / math.sqrt(len(u_set) * len(v_set))
        return W

    # 优化就是那用户之间的相似度
    def userSimilarityBetter(self):
        # 得到每个item被哪些user评价过
        item_users = dict()
        for u, items in self.user_score_dict.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                if self.user_score_dict[u][i] > 0:
                    item_users[i].add(u)
        # 构建倒叙表
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if u == v:
                        continue
                    C[u][v] += 1
        print(C)
        print(N)
        #构建相似度矩阵
        W = dict()
        for u,related_users in C.items():
            W.setdefault(u,{})
            for v,cuv in related_users.items():
                if u == v:
                    continue
                W[u].setdefault(v,0.0)
                W[u][v] = cuv/math.sqrt(N[u]*N[v])
        return W


    def preUserItemScore(self, userA, item):
        score = 0.0
        for user in self.users_sim[userA].keys():
            if user != userA:
                score += self.users_sim[userA][user] * self.user_score_dict[user][item]
        return score

    def recommend(self, userA):
        # 计算userA未评分item的可能性
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            if self.user_score_dict[userA][item] <= 0:
                user_item_score_dict[item] = self.preUserItemScore(userA, item)
        return user_item_score_dict 


if __name__ == "__main__":
    uc = UserCF()
    print(uc.recommend("C"))
