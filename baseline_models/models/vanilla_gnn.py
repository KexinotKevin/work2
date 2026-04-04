import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from layers import relationGCN


class BaseGNN(nn.Module):
    def __init__(self, args, act=F.relu):
        super().__init__()
        self.in_channels = args.input_dimension
        self.hidden_channels = args.hidden_dimension
        self.out_channels = args.output_dimension
        self.drop = args.dropout
        self.depth = args.depth
        self.act = act

        self.lin1 = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.lin2 = nn.Linear(self.hidden_channels, self.hidden_channels // 2)
        self.lin3 = nn.Linear(self.hidden_channels // 2, 1)

    def forward_readout(self, x, batch):
        glob_x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        out_cog = F.leaky_relu(self.lin1(glob_x), negative_slope=0.1)
        out_cog = F.dropout(out_cog, p=self.drop, training=self.training)
        out_cog = F.leaky_relu(self.lin2(out_cog), negative_slope=0.1)
        out_cog = F.dropout(out_cog, p=self.drop, training=self.training)
        return self.lin3(out_cog), None, None


class VanillaGCN(BaseGNN):
    def __init__(self, args):
        super().__init__(args)
        self.convs = nn.ModuleList([GCNConv(self.in_channels, self.hidden_channels)])
        for _ in range(self.depth - 1):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(self.depth)])

    def forward(self, g, label_emb=None, batch=None):
        x, edge_index = g.x, g.edge_index
        batch = edge_index.new_zeros(x.size(0)) if batch is None else batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(self.lns[i](x))
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.forward_readout(x, batch)


class VanillaGAT(BaseGNN):
    def __init__(self, args):
        super().__init__(args)
        heads = 4
        self.convs = nn.ModuleList([GATConv(self.in_channels, self.hidden_channels // heads, heads=heads)])
        for _ in range(self.depth - 1):
            self.convs.append(GATConv(self.hidden_channels, self.hidden_channels // heads, heads=heads))
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(self.depth)])

    def forward(self, g, label_emb=None, batch=None):
        x, edge_index = g.x, g.edge_index
        batch = edge_index.new_zeros(x.size(0)) if batch is None else batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(self.lns[i](x))
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.forward_readout(x, batch)


class VanillaSAGE(BaseGNN):
    def __init__(self, args):
        super().__init__(args)
        self.convs = nn.ModuleList([SAGEConv(self.in_channels, self.hidden_channels)])
        for _ in range(self.depth - 1):
            self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(self.depth)])

    def forward(self, g, label_emb=None, batch=None):
        x, edge_index = g.x, g.edge_index
        batch = edge_index.new_zeros(x.size(0)) if batch is None else batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(self.lns[i](x))
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.forward_readout(x, batch)


class VanillaRelGNN(BaseGNN):
    def __init__(self, args):
        super().__init__(args)
        self.relation_num = args.relation_num
        self.convs = nn.ModuleList([relationGCN(in_dim=self.in_channels, out_dim=self.hidden_channels, relation_num=self.relation_num)])
        for _ in range(self.depth - 1):
            self.convs.append(relationGCN(in_dim=self.hidden_channels, out_dim=self.hidden_channels, relation_num=self.relation_num))
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(self.depth)])

    def forward(self, g, label_emb=None, batch=None):
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
        batch = edge_index.new_zeros(x.size(0)) if batch is None else batch
        for i, conv in enumerate(self.convs):
            x, edge_index, edge_attr = conv(x, edge_index, edge_attr)
            x = self.act(self.lns[i](x))
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.forward_readout(x, batch)
