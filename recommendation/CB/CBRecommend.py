import json
import pandas as pd
import numpy as np
import math
import random


class CBRecommend:
    # 加载dataProcessing中预处理的数据
    def __init__(self, K):
        # 给用户推荐的个数
        self.K = K
        self.item_profile = json.load(open('../../data/ml-1m/use/item_profile.json', 'r'))
        self.user_profile = json.load(open('../../data/ml-1m/use/user_profile.json', 'r'))

    # 获取用户未进行评分的item表
    def get_none_score_item(self, user):
        items = pd.read_csv('../../data/ml-1m/use/movies.csv')['MovieID'].values
        data = pd.read_csv('../../data/ml-1m/use/ratings.csv')
        have_score_items = data[data['UserID'] == user]['MovieID'].values
        none_score_items = set(items) - set(have_score_items)
        return none_score_items

    # 获取用户对item的喜好程度
    def cosUI(self, user, item):
        Uia = sum(
            np.array(self.user_profile[str(user)])
            *
            np.array(self.item_profile[str(item)])
        )

        Ua = math.sqrt(sum([math.pow(one, 2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one, 2) for one in self.item_profile[str(item)]]))
        return Uia/(Ua*Ia)

    #为用户进行电影推荐
    def recommend(self,user):
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user,item)
        if self.K is None:
            result = sorted(
                user_result.items(),key=lambda k:k[1],reverse=True
            )
        else:
            result = sorted(
                user_result.items(),key=lambda k:k[1], reverse=True
            )[:self.K]
        print(result)



if __name__ == '__main__':
    cb = CBRecommend(K=10)
    cb.recommend(1)