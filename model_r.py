import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch_geometric.nn import GCNConv,global_mean_pool
from layers import LGMVPool, GCN, relationGCN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

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

        channels = self.hidden_channels

        self.input_layer = relationGCN(in_dim=self.in_channels,
                                out_dim=self.hidden_channels,
                                relation_num=3)
        self.down_convs = nn.ModuleList()
        # self.pools = nn.ModuleList()
        # self.down_convs.append(GCNConv(self.in_channels, channels))
        for i in range(self.depth):
            # self.pools.append(LGMVPool(channels, self.pool_ratios[i], 0.3))
            self.down_convs.append(relationGCN(in_dim=self.in_channels,out_dim=self.hidden_channels,relation_num=3))

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
        # edge_weight = new_edge_attr[:, 0]
        edge_weight = new_edge_attr
        x = self.act(new_x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        # encoding process
        for i in range(self.depth):
            # x, edge_index, edge_weight, batch, perm = self.pools[i](x, edge_index, edge_weight, label_emb, batch)
            x, edge_index, edge_weight = self.down_convs[i](x, edge_index, edge_weight)
            x = F.dropout(x, training=self.training, p=0.5)
            x = self.act(x)

        xs += [x]
        edge_indices += [edge_index]
        edge_weights += [edge_weight]
            # perms += [perm]

        # readout
        x_cl=[]
        for x in xs:
            glob_x = torch.tensor(torch.cat([gmp(x, batch), gap(x, batch)], dim=1), dtype=x.dtype, device=x.device)
            x_cl.append(F.relu(glob_x))
        x_cl = sum(x_cl)

        # out = torch.tensor(x_cl.view(x_cl.size(0)//self.in_channels, -1), dtype=torch.float64)
        out = F.relu(self.lin1(x_cl))
        out = F.dropout(out, p=self.drop, training=self.training)
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.drop, training=self.training)
        out = F.relu(self.lin3(out))
        # out = F.relu(self.lin4(out.t()))

        # # task1: label prediction
        # graph_emb = global_mean_pool(x, batch)
        # graph_hid = F.relu(self.pred_projection(graph_emb))
        # label_pred = self.cluster_projection(graph_hid)

        # task2: cluster
        # Clustering Assignment (Soft)
        # q = 1.0 / (1.0 + torch.sum((x_cl.unsqueeze(1) - self.cluster_layer) ** 2, dim=2))
        # q = q ** 2 / q.sum(1).view(-1, 1)  # Student's t-distribution
# 
        # task3: reconstruction
        # for i in range(self.depth):
        #     j = self.depth - 1 - i
        
        #     res = xs[j]
        #     edge_index = edge_indices[j]
        #     edge_weight = edge_weights[j]
        #     perm = perms[j]
        
        #     up = torch.zeros_like(res)
        #     up[perm] = x
        #     x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
        #     x, edge_index, edge_weight = self.up_convs[i](x, edge_index, edge_weight)
        #     x = F.dropout(x, training=self.training, p=self.drop)
        #     x = self.act(x)
        # final_x, final_edge_index, final_edge_weight= self.up_convs[-1](x, edge_index, edge_weight)
        #
        return out