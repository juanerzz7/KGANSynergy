import argparse
import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_curve
from tqdm import tqdm  # 产生进度条
import load_gat, dataloader4KGNN
from model import KGANS
import copy
from utils import evaluate
from torch.utils.data import DataLoader
import sklearn.metrics as m
import math


def eval_classification(labels, logits):
    auc = roc_auc_score(y_true=labels, y_score=logits)
    p, r, t = precision_recall_curve(y_true=labels, probas_pred=logits)
    aupr = m.auc(r, p)
    fpr, tpr, threshold = roc_curve(labels, logits)
    # 利用Youden's index计算阈值
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    predicted_label = copy.deepcopy(logits)
    youden_thresh = round(youden_thresh, 3)
    print(youden_thresh)

    predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]
    p_1 = evaluate.precision(y_true=labels, y_pred=predicted_label)
    r_1 = evaluate.recall(y_true=labels, y_pred=predicted_label)
    acc = accuracy_score(y_true=labels, y_pred=predicted_label)
    f1 = f1_score(y_true=labels, y_pred=predicted_label)
    return p_1, r_1, acc, auc, aupr, f1


def train():
    now = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))
    hours = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--CV', type=int, default=1, help='the number of CV')
    parser.add_argument('--n_epochs', type=int, default=1000, help='the number of epochs')
    parser.add_argument('--n_heads', type=int, default=2, help='the number of multi-head')
    parser.add_argument('--n_neighbors', type=int, default=6, help='the number of neighbors to be sampled')

    parser.add_argument('--e_dim', type=int, default=64, help='dimension of user and entity embeddings')
    parser.add_argument('--r_dim', type=int, default=64, help='dimension of user and relation embeddings')
    parser.add_argument('--n_iter', type=int, default=3,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # OS
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--patience', type=int, default=10,
                        help='how long to wait after last time validation loss improved')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # OS
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.01, help='dropout')
    parser.add_argument('--train_test_mode', type=int, default=1, help='Judeg train or test.')

    args = parser.parse_args(['--l2_weight', '1e-4'])

    # filename = '{}_cv{}_dim{}_neig{}_iter{}_heads{}_lr{}_batchsize{}'.format(hours, args.CV, args.e_dim,
    #                                                                          args.n_neighbors, args.n_iter,
    #                                                                          args.n_heads, args.lr, args.batch_size)
    auc_all = []
    aupr_all = []
    acc_all = []
    # f2 = open('result/{}.txt'.format(filename), 'w')
    for cv in range(1):
        auc_cv = []
        aupr_cv = []
        acc_cv = []
        drug1, drug2, cells, triples = load_gat.readRecData()  # 读取药物组合协同数据
        entitys, relations, kgTriples = load_gat.readKGData()  # 读取知识图谱数据
        kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kgTriples)  # 获得三元组

        adj_entity, adj_relation = dataloader4KGNN.construct_adj(args.n_neighbors,
                                                                 kg_indexes, len(entitys))

        loss_fcn = nn.BCELoss()
        np.random.seed(23)
        random.shuffle(triples)
        # split = math.ceil(len(triples) / 5)
        triples_DF = pd.DataFrame(triples)


        test_fold = 0
        for i in range(5):
            auc_kfold = []
            aupr_kfold = []
            acc_kfold = []
            idx_test = np.where(triples_DF[4] == test_fold)
            idx_train = np.where(triples_DF[4] != test_fold)
            test_set = [triples[xx] for xx in idx_test[0]]
            train_set = [triples[xx] for xx in idx_train[0]]
            print(len(train_set) // args.batch_size)

            net = KGANS(args, max(entitys) + 1, max(relations) + 1,
                        args.e_dim, args.r_dim, adj_entity, adj_relation)

            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

            if args.train_test_mode == 1:
                t_total = time.time()
                for e in range(args.n_epochs):
                    t = time.time()
                    net.train()
                    all_loss = 0.0
                    for u1, u2, c, r, fold in DataLoader(train_set, batch_size=args.batch_size,
                                                         shuffle=True):
                        logits = net(u1, u2, c)
                        optimizer.zero_grad()
                        loss = loss_fcn(logits, r.float())
                        loss.backward()
                        optimizer.step()
                        # .item():得到张量里的元素值
                        all_loss += loss.item()
                    loss_train = all_loss / (len(train_set) // args.batch_size)
                    print('[test_fold {},epoch {}],avg_loss={:.4f}'.format(test_fold, e,
                                                                           loss_train))
                    # print('[cv {},flod {},epoch {}],avg_loss={:.4f}'.format(cv + 1, count, e,
                    #                                                         loss_train))
                    # early stop
                    if (e == 0):
                        best_train_loss = loss_train
                        torch.save(net.state_dict(),
                                   'model/{}_decoder{}.pkl'.format(now, test_fold))  # 保存网络中的参数
                        print("save model")
                        earlystop_count = 0

                    else:
                        if best_train_loss > loss_train:
                            best_train_loss = loss_train
                            torch.save(net.state_dict(),
                                       'model/{}_decoder{}.pkl'.format(now, test_fold))  # 保存网络中的参数
                            print("save model")
                            earlystop_count = 0

                        if earlystop_count != args.patience:
                            earlystop_count += 1
                        else:
                            print("early stop!!!!")
                            break

                print("\nOptimization Finished!")
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

            net = KGANS(args, max(entitys) + 1, max(relations) + 1,
                         args.e_dim, args.r_dim, adj_entity, adj_relation)
            net.load_state_dict(torch.load('model/{}_decoder{}.pkl'.format(now, test_fold)))
            test_set = torch.LongTensor(test_set)
            with torch.no_grad():
                net.eval()
                drug1_ids = test_set[:, 0]
                drug2_ids = test_set[:, 1]
                cell_ids = test_set[:, 2]
                labels = test_set[:, 3]
                logits = net(drug1_ids, drug2_ids, cell_ids)

                p, r, acc, auc, aupr, f1 = eval_classification(labels, logits)
                print(
                    'test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f} | auc {:.4f} | aupr {:.4f} | F1 {:.4f}'.format(
                        p, r, acc, auc, aupr, f1))
                auc_kfold.append(auc)
                aupr_kfold.append(aupr)
                acc_kfold.append(acc)
                auc_cv.append(auc)
                aupr_cv.append(aupr)
                acc_cv.append(acc)

            del net
            test_fold += 1
        auc_mean = np.mean(auc_cv)
        aupr_mean = np.mean(aupr_cv)
        acc_mean = np.mean(acc_cv)
        auc_all.append(auc_mean)
        aupr_all.append(aupr_mean)
        acc_all.append(acc_mean)
        print('result: auc {:.4f} | aupr {:.4f} | accuracy {:.4f}'.format(auc_mean, aupr_mean, acc_mean))
        # f2.write('%.6f' % auc_mean + '\t' + '%.6f' % aupr_mean + '\t' + '%.6f' % acc_mean + '\n')

    final_auc = np.mean(auc_all)
    final_aupr = np.mean(aupr_all)
    final_acc = np.mean(acc_all)
    print('Final result: auc {:.4f} | aupr {:.4f} | accuracy {:.4f}'.format(final_auc, final_aupr, final_acc))
    # f2.write('Final result:' + '%.4f' % final_auc + '\t' + '%.4f' % final_aupr + '\t' + '%.4f' % final_acc + '\n')
    # f2.close()


if __name__ == '__main__':
    seed = 55
    random.seed(seed)
    np.random.seed(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()
