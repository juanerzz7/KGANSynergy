import numpy as np
import scipy.sparse as sp
import re
import sys
import torch
import nltk
import math

nltk.download('punkt')


# adapted from pygat
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


# adapted from pygat
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _compact(ent):
    if ' ' in ent:
        ent = ent.replace(' ', '')
    return ent


def _extract(ent):
    ent = str(ent)

    if '#' in ent:
        ent = ent.split('#')[-1]
    else:
        last = ent.split('/')[-1]
        if last == '':
            num = len(ent.split('/')) - 2
            last = ent.split('/')[num]
        ent = last
        if ':' in ent:
            ent = ent.split(':')[-1]
    ent = re.sub('<', '', ent)
    ent = re.sub('>', '', ent)
    ent = re.sub('-', '', ent)
    ent = re.sub('_', '', ent)
    return ent


# Build dictionary word to index
def build_dict(f_path):
    word2ix = {}
    with open(f_path, "r", encoding="utf-8") as f:
        for _, pair in enumerate(f):
            try:
                temp = pair.strip().split("\t")
                word2ix[temp[1]] = int(temp[0])
            except:
                print(temp)
    return word2ix


# Build word to vector
def build_vec(word2ix, word_embedding):
    word2vec = {}
    for word in word2ix:
        word2vec[word] = word_embedding[int(word2ix[word])]
    return word2vec


# Convert to tensor

def tensor_from_data(tensor_concatenation_model, entity_dict, pred_dict, edesc, word_emb, word_emb_calc):
    if tensor_concatenation_model == 1:
        pred_tensor, obj_tensor = tensor_concatenation_model_1(entity_dict, pred_dict, edesc, word_emb, word_emb_calc)
    elif tensor_concatenation_model == 2:
        pred_tensor, obj_tensor = tensor_concatenation_model_2(entity_dict, pred_dict, edesc, word_emb, word_emb_calc)
    elif tensor_concatenation_model == 3:
        pred_tensor, obj_tensor = tensor_concatenation_model_3(entity_dict, pred_dict, edesc, word_emb, word_emb_calc)
    elif tensor_concatenation_model == 4:
        pred_tensor, obj_tensor = tensor_concatenation_model_4(entity_dict, pred_dict, edesc, word_emb, word_emb_calc)
    else:
        print("please choose the correct loss fucntion")
        sys.exit()
    return pred_tensor, obj_tensor


def tensor_concatenation_model_2(entity_dict, pred_dict, edesc, word_emb, word_emb_calc):
    '''
    Tensor concatenation model 2 is obtained by the concatenation of KGE (DistMult/ComplEx) as predicate embeddings and
    the selection object of resources or literals to apply appropriate embedding model as object embeddings.
    If the object is a resource then KGE will be applied to the object. Otherwise, word embedding will be implemented on it.
    '''
    pred_list, obj_list, obj_literal_list, status_list = [], [], [], []
    for _, _, pred, obj, obj_literal, status in edesc:
        pred_list.append(pred_dict[pred])
        obj_list.append(obj)
        obj_literal_list.append(obj_literal)
        status_list.append(status)

    pred_tensor = torch.tensor(pred_list).unsqueeze(1)

    arrays_obj_literal_list = []
    for i, obj in enumerate(obj_literal_list):
        arrays = []
        tokens = nltk.word_tokenize(obj)
        flag = True
        # print(tokens)
        for token in tokens:
            try:
                vec = word_emb[token]
            except:
                vec = np.zeros([300, ])
                flag = False
            arrays.append(vec)
        if word_emb_calc == "SUM":
            obj_vector = np.sum(arrays, axis=0)
        else:
            obj_vector = np.average(arrays, axis=0)

        if flag == False:
            obj_vector = entity_dict[obj_list[i]]
        arrays_obj_literal_list.append(obj_vector)
    obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)

    return pred_tensor, obj_tensor


def tensor_concatenation_model_1(entity_dict, pred_dict, edesc, word_emb, word_emb_calc):
    '''
    Tensor concatenation model 1 is obtained by the concatenation of KGE (DistMult/ComplEx) as predicate embeddings and
    word embeddings as object embeddings
    '''
    # print("#####")
    # print(edesc)
    pred_list, obj_list, obj_literal_list = [], [], []
    for _, _, pred, obj, obj_literal in edesc:
        pred_list.append(pred_dict[pred])
        obj_list.append(obj)
        obj_literal_list.append(obj_literal)

    pred_tensor = torch.tensor(pred_list).unsqueeze(1)

    arrays_obj_literal_list = []
    for obj in obj_literal_list:
        arrays = []
        tokens = nltk.word_tokenize(obj)
        # print("tokens", tokens, obj)
        for token in tokens:
            try:
                vec = word_emb[token]
            except:
                vec = np.zeros([300, ])
                # flag=False
            # print(token, vec, vec.shape)
            arrays.append(vec)
        if len(tokens) > 1:
            if word_emb_calc == "SUM":
                obj_vector = np.sum(arrays, axis=0)
            else:
                obj_vector = np.average(arrays, axis=0)
        else:
            obj_vector = arrays[0]
        # print(obj)
        # print(obj_vector)
        arrays_obj_literal_list.append(obj_vector)
    # arrays_obj_literal_list = np.array(arrays_obj_literal_list)
    # print(arrays_obj_literal_list.shape)
    obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)

    return pred_tensor, obj_tensor


def tensor_concatenation_model_3(entity_dict, pred_dict, edesc, word_emb, word_emb_calc):
    '''
    Tensor concatenation model 3 is obtained by the concatenation of KGE (DistMult/ComplEx) as predicate embeddings and
    the selection object of resources or literals to apply appropriate embedding model as object embeddings.
    If the object is a resource then KGE will be applied to the object. Otherwise, word embedding will be implemented on it.
    '''
    pred_list, obj_list, obj_literal_list, status_list = [], [], [], []
    for _, _, pred, obj, obj_literal, status in edesc:
        pred_list.append(pred_dict[pred])
        obj_list.append(obj)
        obj_literal_list.append(obj_literal)
        status_list.append(status)

    pred_tensor = torch.tensor(pred_list).unsqueeze(1)

    arrays_obj_literal_list = []
    for i, obj in enumerate(obj_literal_list):
        arrays = []
        if status_list[i] == "literal":
            tokens = nltk.word_tokenize(obj_literal_list[i])
            for token in tokens:
                try:
                    vec = word_emb[token]
                except:
                    vec = np.zeros([300, ])
                arrays.append(vec)
            if word_emb_calc == "SUM":
                obj_vector = np.sum(arrays, axis=0)
            else:
                obj_vector = np.average(arrays, axis=0)
        else:
            obj_vector = entity_dict[obj_list[i]]
        arrays_obj_literal_list.append(obj_vector)
    obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)

    return pred_tensor, obj_tensor


def tensor_concatenation_model_4(entity_dict, pred_dict, edesc, word_emb, word_emb_calc):
    '''
    Tensor concatenation model 4 is obtained by the concatenation of KGE (DistMult/ComplEx) as predicate embeddings and
    word embeddings as object embeddings
    '''
    pred_list, obj_list, obj_literal_list, status_list = [], [], [], []
    for _, _, pred, obj, obj_literal, status in edesc:
        pred_list.append(pred_dict[pred])
        obj_list.append(obj)
        obj_literal_list.append(obj_literal)
        status_list.append(status)

    pred_tensor = torch.tensor(pred_list).unsqueeze(1)

    arrays_obj_literal_list = []
    for obj in obj_literal_list:
        vec = np.zeros([300, ])
        obj_vector = vec
        arrays_obj_literal_list.append(obj_vector)
    obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)

    return pred_tensor, obj_tensor


# Define label/target tensor
def tensor_from_weight(tensor_size, edesc, label):
    weight_tensor = torch.zeros(tensor_size)
    for label_word in label:
        order = -1
        for _, _, pred, obj, _ in edesc:
            order += 1
            data_word = "{}++$++{}".format(pred, obj)
            if label_word == data_word:
                weight_tensor[order] += label[label_word]
                break
    return weight_tensor / torch.sum(weight_tensor)


def _eval_Fmeasure(summ_tids, gold_list):
    k = len(summ_tids)
    # print(summ_tids)
    f_list = []
    # print(gold_list)
    for gold in gold_list:
        # print(gold)
        if len(gold) != k:
            print('gold-k:', len(gold), k)
        assert len(gold) == k  # for ESBM
        corr = len([t for t in summ_tids if t in gold])
        # print(corr)
        precision = corr / k
        recall = corr / len(gold)
        f1 = 2 * ((precision * recall) / (precision + recall)) if corr != 0 else 0
        f_list.append(f1)
        # print('corr-prf:',corr,precision,recall,f1)
    favg = np.mean(f_list)
    # print('flist:',favg,f_list)
    return favg