import numpy as np
import scipy.sparse as sp

from utils import osUtils as ou
import sys
import random
import copy


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def readKGData(path='data_set/OS/kg_final2.txt'):
    print('读取知识图谱数据...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in ou.readTriple(path, sep=','):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples


def readRecData(path='data_set/OS/z/rating_final.txt', test_ratio=0.2):
    print('读取药物组合协同数据...')
    user_set1, user_set2, item_set = set(), set(), set()
    triples = []
    for u1, u2, i, r in ou.readTriple(path, sep=','):
        user_set1.add(int(u1))
        user_set2.add(int(u2))
        item_set.add(int(i))
        triples.append((int(u1), int(u2), int(i), int(r)))

    return list(user_set1), list(user_set2), list(item_set), triples


from torch.utils.data import Dataset


# 继承torch自带的Dataset类,重构__getitem__与__len__方法
# class KgDatasetWithNegativeSampling(Dataset):
#
#     def __init__(self, triples, entitys):
#         self.triples = triples  # 知识图谱HRT三元组
#         self.entitys = entitys  # 所有实体集合列表
#
#     def __getitem__(self, index):
#         '''
#         :param index: 一批次采样的列表索引序号
#         '''
#         # 根据索引取出正例
#         pos_triple = self.triples[index]
#         # 通过负例采样的方法得到负例
#         neg_triple = self.negtiveSampling(pos_triple)
#         return pos_triple, neg_triple

    # 负例采样方法
    # def negtiveSampling(self, triple):
    #     seed = random.random()
    #     neg_triple = copy.deepcopy(triple)
    #     if seed > 0.5:  # 替换head
    #         rand_head = triple[0]
    #         while rand_head == triple[0]:  # 如果采样得到自己则继续循环
    #             # 从所有实体中随机采样一个实体
    #             rand_head = random.sample(self.entitys, 1)[0]
    #         neg_triple[0] = rand_head
    #     else:  # 替换tail
    #         rand_tail = triple[2]
    #         while rand_tail == triple[2]:
    #             rand_tail = random.sample(self.entitys, 1)[0]
    #         neg_triple[2] = rand_tail
    #     return neg_triple
    #
    # def __len__(self):
    #     return len(self.triples)


# if __name__ == '__main__':
#     entitys, relations, triples = readKGData()
#     train_set = KgDatasetWithNegativeSampling(triples, entitys)
#     from torch.utils.data import DataLoader
#
#     for set in DataLoader(train_set, batch_size=8, shuffle=True):
#         pos_set, neg_set = set
#         print(pos_set)
#         print(neg_set)
#         sys.exit()
