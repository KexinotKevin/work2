import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsemax import Sparsemax
from torch.nn import Parameter
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add

from selection import *

class relationGCN(MessagePassing):
    def __init__(self, in_dim, out_dim, relation_num, num_bases=-1, bias=True, ifRank=False):
        super(relationGCN, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = relation_num
        self.num_bases = num_bases
        self.ifRank = ifRank

        # 权重初始化
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.w_rel = Parameter(torch.FloatTensor(relation_num, num_bases))
            nn.init.xavier_uniform_(self.w_bases, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))
        else:
            self.w = Parameter(torch.FloatTensor(relation_num, in_dim, out_dim))
            nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_attr):
        # 归一化处理
        edge_index, norm_edge_attr = self.norm(edge_index, x.size(0), edge_attr)

        # 消息传递
        out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)

        # 处理多关系
        for r in range(self.num_rels):
            # 获取当前关系的权重矩阵
            if self.num_bases > 0:
                w = torch.einsum('rb,bio->rio', self.w_rel, self.w_bases).to(x.dtype)
            else:
                w = self.w.to(x.dtype)

            # 创建稀疏邻接矩阵
            adj = torch.sparse_coo_tensor(
                edge_index,
                norm_edge_attr[:, r],
                (x.size(0), x.size(0))
            ).to(x.dtype)

            # 执行矩阵乘法
            support = torch.spmm(adj, x)
            out += torch.matmul(support, w[r])

        # norm_edge_attr = [(edge_attr[k, :] / torch.sum(edge_attr[k, :])).unsqueeze(0) for k in
        #                   range(edge_attr.size(0))]
        # norm_edge_attr = torch.tensor(torch.cat(norm_edge_attr, dim=0), dtype=edge_attr.dtype, device=edge_attr.device)
        # 添加偏置
        if self.bias is not None:
            out += self.bias
        if self.ifRank:
            new_edge_attr = self.process_pagerank(x, edge_index, edge_attr, x.size(0), edge_attr.size(1))
        else:
            new_edge_attr = edge_attr
            # new_edge_attr = self.linear(x)

        return out, edge_index, new_edge_attr

    def process_pagerank(self, x, edge_index, edge_attr, num_nodes, num_rel):
        self.cal_pr = PageRankScore(self.in_dim)
        scores = self.cal_pr(x, edge_index, edge_attr, num_nodes, num_rel)
        max_indices = torch.argmax(scores, dim=1)
        # renew attributes of edge
        row = edge_index[0]
        edge_indices = max_indices[row]
        # construct new edge attr: [score, loc(score)]
        selected_values = edge_attr[torch.arange(edge_attr.size(0)), edge_indices]
        new_attr = torch.stack([selected_values, edge_indices.float()], dim=1)

        return new_attr

    # 优化调整：防止 norm 函数污染原始的边属性，创建一个新的张量来专门存放归一化后的结果即可。
    @staticmethod
    def norm(edge_index, num_nodes, edge_attr, dtype=None):
        row, col = edge_index
        num_relations = edge_attr.size(1)
        # 创建一个形状和类型一致的新张量，防止污染传给下一层的原始特征
        norm_edge_attr = torch.empty_like(edge_attr) 
        
        for r in range(num_relations):
            edge_weight = edge_attr[:, r].clone()
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            norm_edge_attr[:, r] = norm  # 赋值给新张量

        return edge_index, norm_edge_attr

class RGCN(MessagePassing):
    def __init__(self, in_dim, out_dim, relation_num, num_bases=-1, bias=True, ifRank=False):
        super(RGCN, self).__init__(aggr='add')  # 使用 'add' 聚合方式
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = relation_num
        self.num_bases = num_bases
        self.ifRank = ifRank

        # 可学习参数 att_map，用于计算消息传播权重
        self.att_map = Parameter(torch.Tensor(relation_num, 3))  # [relation_num, 3]
        nn.init.xavier_uniform_(self.att_map)  # 初始化 att_map

        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.w_rel = Parameter(torch.FloatTensor(relation_num, num_bases))
            nn.init.xavier_uniform_(self.w_bases, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))
        else:
            self.w = Parameter(torch.FloatTensor(relation_num, in_dim, out_dim))
            nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_attr):
        # 归一化处理
        edge_index, norm_edge_attr = self.norm(edge_index, x.size(0), edge_attr)

        # 消息传递
        out = torch.zeros(x.size(0), self.out_dim, device=x.device)

        # 处理多关系
        for r in range(self.num_rels):
            # 获取当前关系的权重矩阵
            if self.num_bases > 0:
                w = torch.einsum('rb,bio->rio', self.w_rel, self.w_bases)
            else:
                w = self.w
            w = torch.tensor(w, dtype=x.dtype, device=x.device)

            # 计算消息传播权重
            edge_weight = norm_edge_attr[:, r]  # [num_edges, 1]
            att_weights = torch.sum(edge_attr * self.att_map[r], dim=1, keepdim=True).squeeze(1)  # [num_edges, 1]
            edge_weight = edge_weight * att_weights

            # 创建稀疏邻接矩阵
            adj = torch.sparse_coo_tensor(
                edge_index,
                edge_weight.squeeze(),  # 去掉多余的维度
                (x.size(0), x.size(0))
            )

            # 执行矩阵乘法
            support = torch.spmm(adj, x)
            out += torch.matmul(support, w[r])

        # 添加偏置
        if self.bias is not None:
            out += self.bias

        # 处理 PageRank 或直接返回 edge_attr
        if self.ifRank:
            new_edge_attr = self.process_pagerank(x, edge_index, edge_attr, x.size(0), edge_attr.size(1))
        else:
            new_edge_attr = edge_attr

        return out, edge_index, new_edge_attr

    def process_pagerank(self, x, edge_index, edge_attr, num_nodes, num_rel):
        self.cal_pr = PageRankScore(self.in_dim)
        scores = self.cal_pr(x, edge_index, edge_attr, num_nodes, num_rel)
        max_indices = torch.argmax(scores, dim=1)
        # 更新边的属性
        row = edge_index[0]
        edge_indices = max_indices[row]
        # 构造新的边属性: [score, loc(score)]
        selected_values = edge_attr[torch.arange(edge_attr.size(0)), edge_indices]
        new_attr = torch.stack([selected_values, edge_indices.float()], dim=1)

        return new_attr

    @staticmethod
    def norm(edge_index, num_nodes, edge_attr, dtype=None):
        row, col = edge_index
        num_relations = edge_attr.size(1)
        for r in range(num_relations):
            edge_weight = edge_attr[:, r].clone()
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            edge_attr[:, r] = norm

        return edge_index, edge_attr


class GCN(MessagePassing):
    def __init__(self, in_dim, out_dim,
                 cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        else:
            edge_weight = torch.tensor(edge_weight, dtype=x.dtype)
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x = self.propagate(edge_index, x=x, norm=norm)

        return x, edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_dim, self.out_dim)


class PageRankScore(MessagePassing):
    def __init__(self, channels, k=10, alpha=0.1, **kwargs):
        super(PageRankScore, self).__init__(aggr='add', **kwargs)
        self.channels = channels
        self.k = k
        self.alpha = alpha

    def forward(self, x, edge_index, edge_attr, num_nodes, num_relations):
        pagerank_scores = torch.ones(num_nodes, num_relations, device=x.device) / num_nodes
        # calculate PageRank score for every condition
        for r in range(num_relations):
            # 获取当前关系的边权重
            if num_relations == 1:
                edge_wt = edge_attr.clone()
            else:
                edge_wt = edge_attr[:, r]
            edge_index, norm = self.norm(edge_index, num_nodes, edge_wt, dtype=x.dtype)

            for _ in range(self.k):
                out = self.propagate(edge_index, x=pagerank_scores[:, r].view(-1, 1), norm=norm)
                out = out.view(-1)
                # update pagerank
                pagerank_scores[:, r] = out * (1 - self.alpha) + self.alpha / num_nodes

        return pagerank_scores

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        """
            normalization for edge weight
        """
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # 计算度数
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理无穷大值
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # 归一化
        return edge_index, norm


class LGMVPool(nn.Module):
    def __init__(self, in_dim, ratio, lamb=0.3, negative_slop=0.2, useSparse=True):
        super(LGMVPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.sparse = useSparse
        self.negative_slop = negative_slop
        self.sparse_attention = Sparsemax()  # 使用 Sparsemax
        self.lamb = lamb  # 结构学习中的权重参数

        # 可学习参数
        self.alpha = Parameter(torch.Tensor(1))  # 节点度数分数的权重
        self.beta = Parameter(torch.Tensor(1))  # PageRank 分数的权重
        self.theta = Parameter(torch.Tensor(1))
        self.att = Parameter(torch.Tensor(1, self.in_dim * 2))  # 结构学习中的注意力参数
        nn.init.ones_(self.alpha)
        nn.init.ones_(self.beta)
        nn.init.ones_(self.theta)
        nn.init.xavier_uniform_(self.att.data)

    def forward(self, x, edge_index, edge_attr, label, batch=None):
        """
        改进版 MVPool 的前向传播
        :param x: 节点特征，size=[num_nodes, 90]
        :param edge_index: 边索引，size=[2, edge_num * 2]
        :param edge_attr: 边特征，size=[edge_num * 2, 3]
        :param label: 外部输入的标签向量，size=[1, 90]
        :return: 池化后的节点特征、边索引、边属性、批次信息和选择的节点索引
        """
        num_nodes = x.size(0)
        if batch == None:
            batch = edge_index.new_zeros(num_nodes)

        # node selection
        row, _ = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float)  # 节点度数
        score_deg = torch.sigmoid(self.alpha * torch.log(deg + 1e-16)).view(-1, 1)

        # 计算每个关系的 PageRank 分数并求平均
        pagerank_scores = []
        for i in range(edge_attr.size(1)):  # 遍历每个关系
            edge_weight = edge_attr[:, i]
            pagerank_score = self._calc_pagerank(x, edge_index, edge_weight, x.size(0))
            pagerank_scores.append(pagerank_score)
        pagerank_scores = torch.stack(pagerank_scores, dim=1)
        pagerank_score = torch.mean(pagerank_scores, dim=1)  # 对每个关系的 PageRank 分数求平均
        score_pagerank = torch.sigmoid(self.beta * pagerank_score).view(-1, 1)

        score = (score_deg + score_pagerank) / 2

        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # edge learning
        new_edge_index, new_edge_attr = add_remaining_self_loops(induced_edge_index, induced_edge_attr, 0, x.size(0))
        
        # 【新增修复】显式对边进行排序，满足 sparsemax 内部对 batch(row) 连续递增的严格依赖
        sort_idx = torch.argsort(new_edge_index[0])
        new_edge_index = new_edge_index[:, sort_idx]
        new_edge_attr = new_edge_attr[sort_idx]
        
        row, col = new_edge_index
        # 对每个关系单独计算权重
        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
        # print(weights.size())
        # norm_edge_attr = [new_edge_attr[k, :]/torch.sum(new_edge_attr[k, :]) for k in range(new_edge_attr.size(0))]
        # weights = torch.tensor([torch.matmul(new_edge_attr[k, :], new_edge_attr[k, :].t()) for k in range(new_edge_attr.size(0))], dtype=x.dtype, device=x.device)
        new_attr_list = []
        for i in range(new_edge_attr.size(1)):
            tmp_wt = new_edge_attr[:, i]
            tmp_act_wt = self.lamb * tmp_wt + F.leaky_relu(weights, self.negative_slop)
            
            # 1. 使用 .clone() 避免 Sparsemax 内部的 in-place 操作报错
            # 2. 使用 .to(...) 替代 torch.tensor(...) 保持梯度图连接，消灭 Warning
            sparsed_attr = self.sparse_attention(tmp_act_wt.clone(), row).to(tmp_act_wt.dtype)
            new_attr_list.append(sparsed_attr)

        new_edge_attr = torch.stack(new_attr_list, dim=1)

        # 过滤掉 relation 0 中权重为 0 的边 (与原版 adj[:, :, 0] != 0 的语义严格保持一致)
        mask = new_edge_attr[:, 0] != 0
        new_edge_index = new_edge_index[:, mask]
        new_edge_attr = new_edge_attr[mask]

        return x, new_edge_index, new_edge_attr, perm, batch

    def _calc_pagerank(self, x, edge_index, edge_weight, num_nodes):
        self.cal_pr = PageRankScore(channels=x.size(0))
        return self.cal_pr(x, edge_index, edge_weight, num_nodes, num_relations=1)