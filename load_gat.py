import numpy as np
import scipy.sparse as sp

from utils import osUtils as ou
import sys
import random
import copy


def normalize_adj(mx):
    """Row Normalized Sparse Matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def readKGData(path='data_set/OS/kg_final2.txt'):
    print('Read knowledge graph data...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in ou.readTriple(path, sep=','):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples


def readRecData(path='data_set/OS/comb_final.txt', test_ratio=0.2):
    print('Read Drug Combination Synergy Data...')
    drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, i, r, flod in ou.readTriple(path, sep=','):
        drug_set1.add(int(d1))
        drug_set2.add(int(d2))
        cell_set.add(int(i))
        triples.append((int(d1), int(d2), int(i), int(r), int(flod)))

    return list(drug_set1), list(drug_set2), list(cell_set), triples

