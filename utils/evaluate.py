from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np
import sys


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def accuracy(y_true, y_scores):
    return accuracy_score(y_true, y_scores)


def MSE(y_true, y_pred):
    return np.average((np.array(y_true) - np.array(y_pred)) ** 2)


def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5


def MAE(y_true, y_pred):
    return np.average(abs(np.array(y_true) - np.array(y_pred)))


class TopK_evaluate():

    @staticmethod
    def precisionAndRecall(pred, t_pos, t_neg):
        TP = len(set(pred) & set(t_pos))
        FP = len(set(pred) & set(t_neg))
        all_pos = len(pred)
        all_recall = len(t_pos)
        p = TP / (TP + FP) if TP + FP > 0 else None
        p_full = TP / all_pos
        r = TP / all_recall if all_recall > 0 else None
        return p, p_full, r

    # 覆盖率
    @staticmethod
    def coverage(all_pred, all_items):
        a = set()
        for pred in all_pred:
            a |= set(pred)
        return len(a) / len(all_items)

    # 多样性
    @staticmethod
    def diversity(all_pred):
        a = set()
        for pred in all_pred:
            a |= set(pred)
        return len(a) / (len(all_pred) * len(all_pred[0]))

    @staticmethod
    def hit_rate_for_item(t_items, p_items):
        return len((set(p_items) & set(t_items))) / len(t_items)

    @staticmethod
    def hit_rate_for_user(test_user_item_list, user_recommadations):
        hit = 0
        for user in user_recommadations:
            if len(set(user_recommadations[user]) & set(test_user_item_list[user])) != 0:
                hit += 1
        return hit / len(user_recommadations)

    @staticmethod
    def AP(t_items, p_items):
        hits = 0
        sum_precs = 0
        for n in range(len(p_items)):
            if p_items[n] in t_items:
                hits += 1
                sum_precs += hits / (n + 1.0)
        if hits > 0:
            return sum_precs / len(t_items)
        else:
            return 0

    @staticmethod
    def MAP(test_user_item_list, user_recommadations):
        ap = 0
        for user in user_recommadations:
            ap += TopK_evaluate.AP(test_user_item_list[user], user_recommadations[user])
        return ap / len(user_recommadations)

    @staticmethod
    def RR(t_items, p_items):
        # "Reciprocal Rank"
        for n in range(len(p_items)):
            if p_items[n] in t_items:
                return 1 / n + 1
        else:
            return 0

    @staticmethod
    def MRR(test_user_item_list, user_recommadations):
        rr = 0
        for user in user_recommadations:
            rr += TopK_evaluate.RR(test_user_item_list[user], user_recommadations[user])
        return rr / len(user_recommadations)


if __name__ == '__main__':
    a = [1, 1, 1, 1, 0, 0]
    b = [0, 1, 1, 1, 1, 3]
    print(MSE(a, b))
    print(RMSE(a, b))
    print(MAE(a, b))

    sys.exit()
    t = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14]
    p = [2, 3, 7, 8, 9]

    print(precision(a, b))
    print(recall_score(a, b))
    print(roc_auc_score(a, b))
    print(accuracy(a, b))
