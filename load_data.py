import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os.path as osp
import torch
from torch_geometric.data import data as D
import os
        
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


def _resolve_conn_file(base_path):
    if osp.isfile(base_path):
        return base_path
    csv_path = base_path + ".csv"
    if osp.isfile(csv_path):
        return csv_path
    return None


def _torch_load_graph_cache(path):
    """Load PyG Data dict; PyTorch>=2.4 defaults weights_only=True which breaks PyG objects."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_data(
    netDir,
    subjlist,
    netname,
    labelfile,
    labeltype,
    labeldim,
    Isheader,
    ifBucket=False,
    sc_netnames=None,
    fc_netname=None,
    subject_col="Subject",
    use_cfg_layout=False,
    atlas_name=None,
):
    if sc_netnames is None or fc_netname is None:
        sc_fod = netname[0]
        fc_fod = netname[1]
        sc_netnames = [sc_fod]
        fc_netname = fc_fod

    # ==================== 【核心新增 1：初始化磁盘缓存机制】 ====================
    atlas_tag = atlas_name if atlas_name else "default"
    sc_tag = "-".join(sc_netnames)
    cache_filename = f"cached_graphs_{atlas_tag}_{sc_tag}_{fc_netname}.pt".replace("/", "_")
    cache_path = osp.join(netDir, cache_filename)

    graph_dict = {}
    if osp.exists(cache_path):
        print(f">>> Loading cached graph structures from {cache_path} (Super Fast!)...")
        graph_dict = _torch_load_graph_cache(cache_path)
    else:
        print(">>> No cache found. Parsing massive raw CSVs (This will only happen ONCE)...")
    new_graphs_processed = False
    # =========================================================================

    sc_channel_count = len(sc_netnames)
    graph_list = []
    dt = pd.read_csv(labelfile)
    lb_map = dt[labeltype].values
    subjlist_set = set([str(s) for s in subjlist if str(s)])

    for k in range(dt.shape[0]):
        subj = str(dt[subject_col][k])
        if subj in subjlist_set:
            
            # ==================== 【核心新增 2：命中缓存则跳过 CSV 解析】 ====================
            if subj in graph_dict:
                g_data = graph_dict[subj]
            else:
                # ------ 下面完全是你原封不动的 CSV 解析和图构建逻辑 ------
                if use_cfg_layout:
                    if not atlas_name:
                        raise ValueError("atlas_name is required when use_cfg_layout=True.")
                    sc_mats = []
                    missing_sc = False
                    for sc_name in sc_netnames:
                        sc_base = osp.join(netDir, atlas_name, subj, "SC", sc_name)
                        sc_path = _resolve_conn_file(sc_base)
                        if sc_path is None:
                            missing_sc = True
                            break
                        sc_mats.append(load_connectivity_matrix(sc_path, isheader=True))

                    fc_base = osp.join(netDir, atlas_name, subj, "FC", fc_netname)
                    fc_path = _resolve_conn_file(fc_base)
                    if missing_sc or fc_path is None:
                        continue
                    fc_mat = load_connectivity_matrix(fc_path, isheader=True)
                else:
                    matname = '{}.csv'.format(subj)
                    sc_mat = load_connectivity_matrix(osp.join(netDir, sc_netnames[0], matname))
                    sc_mats = [sc_mat]
                    fc_mat = load_connectivity_matrix(osp.join(netDir, fc_netname, matname), isheader=Isheader)
                
                feat = torch.tensor(get_node_feature(len(fc_mat[0])))
                edge_in = []
                edge_out = []
                edge_attr_l = []

                for i in range(len(fc_mat[0])):
                    for j in range(i, len(fc_mat[0])):
                        edge_in.append(i)
                        edge_out.append(j)
                        edge_attr = []
                        for sc_mat_item in sc_mats:
                            edge_attr.append(sc_mat_item[i][j] if i != j else 0)
                        edge_attr.append(fc_mat[i][j] if fc_mat[i][j]>0 else 0)
                        edge_attr.append(abs(fc_mat[i][j]) if fc_mat[i][j]<0 else 0)
                        edge_attr_l.append(edge_attr)
                edge_attr = torch.tensor(edge_attr_l, dtype=feat.dtype, device=feat.device)
                
                # normalization
                for i in range(len(edge_attr_l[0])):
                    denom = torch.sum(edge_attr[:, i])
                    if denom > 0:
                        edge_attr[:, i] = edge_attr[:, i] / denom
                
                g_data = D.Data()
                g_data.x, g_data.edge_index, g_data.edge_attr = feat, torch.tensor([edge_in, edge_out]), edge_attr
                # ------ 原有 CSV 解析逻辑结束 ------
                
                # 将处理好的纯结构图存入字典
                graph_dict[subj] = g_data
                new_graphs_processed = True
            # =========================================================================

            # 【动态分配当前任务的 Label】（无论来自缓存还是刚处理完，都在这里挂载标签）
            lb = torch.tensor(lb_map[k], dtype=g_data.x.dtype, device=g_data.x.device)
            if ifBucket:
                graph_list.append((g_data, torch.Tensor(label_bucketization(lb))))
            else:
                graph_list.append((g_data, lb))

    # ==================== 【核心新增 3：将新生成的图字典落盘】 ====================
    if new_graphs_processed:
        print(f">>> Saving processed graph caches to {cache_path} ...")
        torch.save(graph_dict, cache_path)
    # =========================================================================

    print("dataset size: {} subjects".format(len(graph_list)))
    return graph_list
