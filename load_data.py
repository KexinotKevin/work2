import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os.path as osp
import torch
from torch_geometric.data import data as D
        
def load_connectivity_matrix(filename, isheader=False):
    if isheader:
        mat = pd.read_csv(filename, header=None).values
    else:
        mat = pd.read_csv(filename).values
    return mat

def get_node_feature(dim):
    return sp.identity(dim).toarray()

def label_bucketization(value):
    buckets = [(0, 90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, float('inf'))]
    one_hot_vector = np.zeros(len(buckets))

    for index, (lower, upper) in enumerate(buckets):
        if lower < value < upper:
            one_hot_vector[index] = 1
            break
        elif value >= 150:
            one_hot_vector[-1] = 1
            break
    return one_hot_vector

def label_encoding(lb_map, dim=90):
    length=len(lb_map)
    # 初始化位置编码矩阵
    pos_enc = np.zeros((length, dim))
    # 生成位置编码
    for pos in range(length):
        for i in range(0, dim, 2):
            # 正弦函数编码
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / dim)))
            # 余弦函数编码
            if i + 1 < dim:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / dim)))
    sorted_score = sorted(lb_map)
    encoded_lb = torch.full((878, dim), 1/dim)
    for i in range(length):
        pos_idx = sorted_score.index(lb_map[i])
        normed_pos_emb = pos_enc[pos_idx] / np.linalg.norm(pos_enc[pos_idx])
        encoded_lb[i, :] = torch.Tensor([lb_map[i]*item for item in normed_pos_emb])
    return encoded_lb

from sklearn.preprocessing import normalize
def load_data(netDir, subjlist, netname, labelfile, labeltype, labeldim, Isheader, ifBucket=False):
    sc_fod, fc_fod = netname[0], netname[1]
    graph_list = []
    dt = pd.read_csv(labelfile)
    lb_map = dt[labeltype].values

    cnt=0
    for k in range(dt.shape[0]):
        subj = dt.Subject[k]
        if subj in subjlist:
            matname = '{}.csv'.format(subj)
            sc_mat = load_connectivity_matrix(osp.join(netDir, sc_fod, matname))
            fc_mat = load_connectivity_matrix(osp.join(netDir, fc_fod, matname), isheader=Isheader)
            # G = nx.MultiGraph()
            feat = torch.tensor(get_node_feature(len(fc_mat[0])))
            edge_in = []
            edge_out = []
            edge_attr_l = []

            # encoded_lb_map = label_encoding(dt[labeltype].values, dim=labeldim)

            for i in range(len(fc_mat[0])):
                # G.add_node(i)
                # G.nodes[i]['feature'] = feat[i, :]
                for j in range(i, len(fc_mat[0])):
                    edge_in.append(i)
                    # edge_in.append(j)
                    edge_out.append(j)
                    # edge_out.append(i)
                    edge_attr = []
                    edge_attr.append(sc_mat[i][j] if i!=j else edge_attr.append(0))
                    edge_attr.append(fc_mat[i][j] if fc_mat[i][j]>0 else 0)
                    edge_attr.append(abs(fc_mat[i][j]) if fc_mat[i][j]<0 else 0)
                    # G.add_edges_from([(i, j, {"sc": np.log(1+sc_mat[i][j])/(1+np.log(1+sc_mat[i][j]))}),
                    #                   (i, j, {"fc_pos": fc_mat[i][j] if fc_mat[i][j]>0 else 0}),
                    #                   (i, j, {"fc_neg": abs(fc_mat[i][j]) if fc_mat[i][j]<0 else 0})])
                    # for k in range(2):
                    edge_attr_l.append(edge_attr)
            edge_attr = torch.tensor(edge_attr_l, dtype=feat.dtype, device=feat.device)
            # normalization
            for i in range(len(edge_attr_l[0])):
                edge_attr[:, 0] = edge_attr[: ,i]/torch.sum(edge_attr[: ,i])
            g_data = D.Data()
            g_data.x, g_data.edge_index, g_data.edge_attr = feat, torch.tensor([edge_in, edge_out]), edge_attr
            # lb = encoded_lb_map[dt[dt['Subject'] == subj].index[0], :]
            lb = torch.tensor(lb_map[k], dtype=feat.dtype, device=feat.device)
            if ifBucket:
                graph_list.append((g_data, torch.Tensor(label_bucketization(lb))))
            else:
                graph_list.append((g_data, lb))
        # cnt+=1
        # if cnt==50:
        #     break
    print("dataset size: {} subjects".format(len(graph_list)))
    return graph_list