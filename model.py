import torch.nn.functional as F
import torch
import torch.nn as nn
import math


class KGANS(nn.Module):

    def  __init__(self,args, n_entitys, n_relations, e_dim, r_dim,
                 adj_entity, adj_relation,agg_method='Bi-Interaction',):
        super(KGANS, self).__init__()

        self.entity_embs = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)
        self.dropout = args.dropout
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.WW = nn.Linear(256, 128, bias=False)
        self.n_heads = args.n_heads
        self.adj_entity = adj_entity  # 节点的邻接列表
        self.adj_relation = adj_relation  # 关系的邻接列表
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weight()

        self.dropout_layer = nn.Dropout(self.dropout)

        self.agg_method = agg_method  # 聚合方法
        self.agg = 'concat'

        # 初始化最终聚合时所用的激活函数
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        # 初始化各种聚合时所用的线性层
        if agg_method == 'concat':
            self.W_concat = nn.Linear(e_dim * 2, e_dim)
        else:
            self.W1 = nn.Linear(e_dim * self.n_heads, e_dim * 2)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim * self.n_heads, e_dim * 2)

    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)

        # init attention
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


    # 得到邻居的节点embedding和关系embedding
    def get_neighbors_cell(self, cells):
        entities = [cells]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_entity[entities[h]]).view((entities[h].shape[0], -1)).to(
                'cpu')
            neighbor_relations = torch.LongTensor(self.adj_relation[entities[h]]).view((entities[h].shape[0], -1)).to(
                'cpu')
            # e_ids = [self.adj_entity[cell] for cell in cells]
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
        return neighbor_entities_embs, neighbor_relations_embs

    def get_neighbors_drug(self, drugs):
        entities = [drugs]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_entity[entities[h]]).view((entities[h].shape[0], -1)).to(
                'cpu')
            neighbor_relations = torch.LongTensor(self.adj_relation[entities[h]]).view((entities[h].shape[0], -1)).to(
                'cpu')
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
        return neighbor_entities_embs, neighbor_relations_embs

    def sum_aggregator(self, embs):
    # 最终层聚合
        e_u = embs[0]
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)

        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += self.WW(embs[i])

        return e_u

    # 利用 multi-head GAT消息传递
    def GATMessagePass(self, h_embs, r_embs, t_embs):
        '''
        :param h_embs: 头实体向量[ batch_size, e_dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, r_dim ]
        :param t_embs: 尾实体向量[ batch_size, n_neibours, e_dim ]
        '''
        muti = []
        for i in range(self.n_heads):
            # 先拼接头实体和关系实体，乘以尾实体 计算知识图谱注意力
            att_weights = self.attention(torch.cat((h_embs, r_embs), dim=-1)).squeeze(-1)
            # [batch_size, triple_set_size]
            att_weights_norm = F.softmax(att_weights, dim=-1)
            # [batch_size, triple_set_size, dim]
            emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_embs)
            # [batch_size, dim]
            Wx = nn.Linear(self.dim, self.dim)
            # 每次改变w
            emb_i = Wx(emb_i.sum(dim=1))
            muti.append(emb_i)
        all_emb_i = torch.cat([c for c in muti], dim=-1)
        return all_emb_i

    # 消息聚合
    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        '''
        :param h_embs: 原始的头实体向量 [ batch_size, e_dim ]
        :param Nh_embs: 消息传递后头实体位置的向量 [ batch_size, e_dim ]
        :param agg_method: 聚合方式，总共有三种,分别是'Bi-Interaction','concat','sum'
        '''
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) \
                   + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:  # sum
            return self.leakyRelu(self.W1(h_embs + Nh_embs))

    def forward(self, u1, u2, c):
        # cell line嵌入学习
        t_embs, r_embs = self.get_neighbors_cell(c)
        h_embs = self.entity_embs(c)
        t_vectors_next_iter = [h_embs]
        for i in range(self.n_iter):
            if i == 0:
                #改变头实体维度
                h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                # 多头注意力
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                h_embs = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                # 聚合
                cell_embs_1 = self.aggregate(h_embs, vector, self.agg_method)  # (64,64)
                t_vectors_next_iter.append(cell_embs_1)
            else:
                h_broadcast_embs = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs[i], dim=1), 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                embs = torch.cat([torch.sum(t_embs[i], dim=1) for _ in range(self.n_heads)], dim=1)
                cell_embs_1 = self.aggregate(embs, vector, self.agg_method)
                t_vectors_next_iter.append(cell_embs_1)
        Nh_embs_list = t_vectors_next_iter
        # self.cell_embs = torch.cat([att for att in Nh_embs_list], dim=1)
        self.cell_embs = self.sum_aggregator(Nh_embs_list)
        # # [ batch_size, n_neibours, e_dim ] and # [ batch_size, n_neibours, r_dim ]

        # drug2嵌入学习过程
        t_embs_drug2, r_embs_drug2 = self.get_neighbors_drug(u2)
        # # [ batch_size, e_dim ]
        h_embs_drug2 = self.entity_embs(u2)
        drug2_t_vectors_next_iter = [h_embs_drug2]
        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug2 = torch.cat(
                    [torch.unsqueeze(h_embs_drug2, 1) for _ in range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug2 = torch.cat([h_embs_drug2 for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(h_embs_drug2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)
            else:
                h_broadcast_embs_drug2 = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs_drug2[i], dim=1), 1) for _ in
                     range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                embs_d2 = torch.cat([torch.sum(t_embs_drug2[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(embs_d2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)
        Nh_embs_drug2_list = drug2_t_vectors_next_iter
        # self.drug2_embs = torch.cat([att for att in Nh_embs_drug2_list], dim=1)
        self.drug2_embs = self.sum_aggregator(Nh_embs_drug2_list)
        # # [ batch_size, e_dim ]

        # drug1嵌入学习过程
        t_embs_drug1, r_embs_drug1 = self.get_neighbors_drug(u1)
        # # [ batch_size, e_dim ]
        h_embs_drug1 = self.entity_embs(u1)
        drug1_t_vectors_next_iter = [h_embs_drug1]
        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug1 = torch.cat(
                    [torch.unsqueeze(h_embs_drug1, 1) for _ in range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug1 = torch.cat([h_embs_drug1 for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(h_embs_drug1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)
            else:
                h_broadcast_embs_drug1 = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs_drug1[i], dim=1), 1) for _ in
                     range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                embs_d1 = torch.cat([torch.sum(t_embs_drug1[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(embs_d1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)
        Nh_embs_drug1_list = drug1_t_vectors_next_iter
        drug1_embs = self.sum_aggregator(Nh_embs_drug1_list)

        combine_drug = torch.max(drug1_embs, self.drug2_embs)
        logits = torch.sigmoid((combine_drug * self.cell_embs).sum(dim=1))
        return logits
