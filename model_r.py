import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch_geometric.nn import GCNConv,global_mean_pool
from layers import LGMVPool, GCN, relationGCN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层：前向传播恒等映射，反向传播时反转梯度方向"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class LGUNet_rela(torch.nn.Module):
    def __init__(self, args, sum_res=False, act=F.relu):
        super(LGUNet_rela, self).__init__()
        assert args.depth >= 1
        self.in_channels = args.input_dimension
        self.hidden_channels = args.hidden_dimension
        self.out_channels = args.output_dimension
        self.drop = args.dropout
        self.depth = args.depth
        self.pool_ratios = args.pool_ratio
        self.act = act
        self.sum_res = sum_res
        self.batch = args.batch

        self.lin1 = torch.nn.Linear(self.hidden_channels*2, self.hidden_channels)
        self.lin2 = torch.nn.Linear(self.hidden_channels, self.hidden_channels // 2)
        self.lin3 = torch.nn.Linear(self.hidden_channels // 2, 1)

        # ======== 修改开始 ========
        # GRL 对抗分支：添加外部开关
        self.use_grl = not getattr(args, 'disable_grl', False)
        
        if self.use_grl:
            self.age_predictor = nn.Sequential(
                nn.Linear(self.hidden_channels * 2, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=self.drop),
                nn.Linear(self.hidden_channels, 1)
            )
            # GRL 对抗分支：性别分类器
            self.gender_predictor = nn.Sequential(
                nn.Linear(self.hidden_channels * 2, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=self.drop),
                nn.Linear(self.hidden_channels, 1)
            )
        # ======== 修改结束 ========

        channels = self.hidden_channels

        self.relation_num = int(getattr(args, "relation_num", 3))
        self.input_layer = relationGCN(in_dim=self.in_channels,
                                out_dim=self.hidden_channels,
                                relation_num=self.relation_num)
        
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # ======== 修改开始 ========
        self.disable_pr = getattr(args, 'disable_pr', False)
        self.disable_el = getattr(args, 'disable_el', False)
        # ======== 修改结束 ========
        
        for i in range(self.depth):
            # 将消融参数传入 LGMVPool
            self.pools.append(LGMVPool(channels, self.pool_ratios[i], 0.3, 
                                       disable_pr=self.disable_pr, 
                                       disable_el=self.disable_el))
            self.down_convs.append(relationGCN(in_dim=self.hidden_channels, out_dim=self.hidden_channels, relation_num=self.relation_num))

        # 【额外修复】：LayerNorm 稳定特征尺度（在特征维度归一化，不破坏图均值）
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(self.depth)])

        # build reconstruction
        # in_channels = channels if sum_res else 2 * channels
        # self.up_convs = nn.ModuleList()
        # for i in range(self.depth):
        #     self.up_convs.append(relationGCN(in_channels, channels, relation_num=3))
        # self.up_convs.append(relationGCN(channels, self.in_channels, relation_num=3))

    def forward(self, g, label_emb, batch=None):
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_attr is None:
            edge_attr = x.new_ones(edge_index.size(1))

        new_x, edge_index, new_edge_attr = self.input_layer(x, edge_index, edge_attr)
        edge_weight = new_edge_attr
        
        x = self.act(new_x)  # 激活

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        # encoding process
        for i in range(self.depth):
            x, edge_index, edge_weight, perm, batch = self.pools[i](x, edge_index, edge_weight, label_emb, batch)
            x, edge_index, edge_weight = self.down_convs[i](x, edge_index, edge_weight)
            
            x = self.lns[i](x)  # 【额外修复】：LayerNorm 稳定特征尺度
            x = F.dropout(x, training=self.training, p=0.5)  # Dropout
            x = self.act(x)  # 激活

        xs += [x]
        edge_indices += [edge_index]
        edge_weights += [edge_weight]
            # perms += [perm]

        # readout
        x_cl=[]
        for x in xs:
            glob_x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            # 【修改】：直接将图向量 append，移除 F.relu，防止特征被截断为空
            x_cl.append(glob_x)
        x_cl = sum(x_cl)

        # 主任务：认知能力预测
        out_cog = F.leaky_relu(self.lin1(x_cl),negative_slope=0.1)
        out_cog = F.dropout(out_cog, p=self.drop, training=self.training)
        out_cog = F.leaky_relu(self.lin2(out_cog),negative_slope=0.1)
        out_cog = F.dropout(out_cog, p=self.drop, training=self.training)
        out_cog = self.lin3(out_cog)

        # ======== 修改开始 ========
        # 对抗任务：年龄与性别预测（通过 GRL 反转梯度）
        if self.use_grl:
            x_reversed = GradientReversalLayer.apply(x_cl, 1.0)
            out_age = self.age_predictor(x_reversed)
            out_gender = self.gender_predictor(x_reversed)
        else:
            out_age, out_gender = None, None
        # ======== 修改结束 ========

        self.saved_edge_weights = edge_weights
        return out_cog, out_age, out_gender