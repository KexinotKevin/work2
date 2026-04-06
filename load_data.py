import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import os.path as osp
import torch
from torch_geometric.data import data as D
import os
from threshold_func import threshold_consistency
        
def load_connectivity_matrix(filename, isheader=False):
    try:
        if isheader:
            mat = pd.read_csv(filename, header=None).values
        else:
            mat = pd.read_csv(filename).values
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(
            f"Connectivity CSV is empty or has no parseable columns: {osp.abspath(filename)}"
        ) from e
    return mat

def get_node_feature(num_nodes,max_nodes=300):
    """
    【核心修复】：使用固定维度 (300) 的 One-hot 编码。
    1. 赋予每个脑区独立的身份标识，打破 GNN 的节点同质化。
    2. 固定特征维度为 300，完美解决不同被试节点数 (N) 不同导致的 PyG Batch 拼接报错。
    """
    feat = np.zeros((num_nodes, max_nodes), dtype=np.float64)
    for i in range(num_nodes):
        if i < max_nodes:
            feat[i, i] = 1.0  # 对角线置 1，形成身份编码
    return feat

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
    """Resolve connectivity file path; ignore missing or zero-byte placeholders."""

    def _usable(path):
        try:
            return osp.isfile(path) and osp.getsize(path) > 0
        except OSError:
            return False

    if _usable(base_path):
        return base_path
    csv_path = base_path + ".csv"
    if _usable(csv_path):
        return csv_path
    return None


def _torch_load_graph_cache(path):
    """Load PyG Data dict; PyTorch>=2.4 defaults weights_only=True which breaks PyG objects."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_mat_file(filepath, key=None):
    """Load .mat file using scipy.io and return the connectivity matrix.
    
    Args:
        filepath: Path to the .mat file
        key: Specific key to extract. If None, uses 'probmat' or first non-meta key.
    """
    try:
        mat = sio.loadmat(filepath)
        if key and key in mat:
            return mat[key]
        if 'probmat' in mat:
            return mat['probmat']
        # Fallback: return first non-meta key
        for k in mat.keys():
            if not k.startswith('__'):
                return mat[k]
        raise ValueError(f"No usable matrix found in {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load MAT file {filepath}: {e}") from e


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
    gender_col=None,
    age_col=None,
    output_dir=None,
    matDir=None,
    use_mat_format=False,
):
    def process_age(val):
        """处理异质性年龄格式：S1200区间型、ABCD整型、HCD连续型"""
        s = str(val).strip()
        try:
            if '-' in s:
                low, high = s.split('-')
                return (float(low) + float(high)) / 2.0
            if '+' in s:
                return float(s.replace('+', '')) + 1.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def process_gender(val):
        """处理异质性性别格式：统一映射为 0(Female) 或 1(Male)"""
        s = str(val).upper().strip()
        if s in ['M', 'MALE', '1', '1.0', 'MALE/M']:
            return 1.0
        return 0.0
    if sc_netnames is None or fc_netname is None:
        sc_fod = netname[0]
        fc_fod = netname[1]
        sc_netnames = [sc_fod]
        fc_netname = fc_fod

    # ==================== 【核心新增 1：初始化磁盘缓存机制】 ====================
    atlas_tag = atlas_name if atlas_name else "default"
    sc_tag = "-".join(sc_netnames)
    # Bump cache tag when node feature layout changes (old caches used N×N identity, now uses fixed 300-dim one-hot).
    cache_filename = f"cached_graphs_{atlas_tag}_{sc_tag}_{fc_netname}_thr75_feat300.pt".replace("/", "_")
    cache_path = osp.join(netDir, cache_filename)

    graph_dict = {}
    if osp.exists(cache_path):
        print(f">>> Loading cached graph structures from {cache_path} (Super Fast!)...")
        graph_dict = _torch_load_graph_cache(cache_path)
        # for debug
        # graph_dict = dict(sorted(graph_dict.items(), key=lambda x: x[0], reverse=True)[:300])
    else:
        print(">>> No cache found. Parsing massive raw CSVs (This will only happen ONCE)...")
    new_graphs_processed = False

    sc_channel_count = len(sc_netnames)
    graph_list = []
    dt = pd.read_csv(labelfile)
    lb_map = dt[labeltype].values
    skipped_bad_label = 0
    subjlist_set = set([str(s) for s in subjlist if str(s)])

    # 预处理年龄与性别映射（用于 GRL 对抗训练）
    age_map = {}
    gender_map = {}
    if age_col and age_col in dt.columns:
        for _, row in dt.iterrows():
            subj = str(row[subject_col])
            age_map[subj] = process_age(row[age_col])
    if gender_col and gender_col in dt.columns:
        for _, row in dt.iterrows():
            subj = str(row[subject_col])
            gender_map[subj] = process_gender(row[gender_col])

    # ==================== 【核心新增：两阶段图构建与一致性阈值过滤】 ====================
    if not graph_dict:
        raw_mats_list = []
        # --- 阶段 1：加载全组受试者的原始矩阵 ---
        for k in range(dt.shape[0]):
            subj = str(dt[subject_col][k])
            if subj in subjlist_set:
                # ABCD 等数据：磁盘目录与分数表中的 src_subject_id 一致（如 NDAR_INV...），
                # 不要去掉 NDAR_ 后的下划线，否则路径指向不存在的 NDARINV... 目录。
                subj_for_file = subj

                # 读取矩阵的原始逻辑
                if use_cfg_layout:
                    if not atlas_name:
                        raise ValueError("atlas_name is required when use_cfg_layout=True.")
                    
                    # ====== 【HCD .mat格式处理 - 新结构】 ======
                    if use_mat_format and matDir:
                        # 新结构: {matDir}/{subj}/SC/pSC.mat 和 FC/pFC.mat
                        sc_path = osp.join(matDir, subj_for_file, "SC", "pSC.mat")
                        fc_path = osp.join(matDir, subj_for_file, "FC", "pFC.mat")
                        if osp.exists(sc_path) and osp.exists(fc_path):
                            sc_mat = load_mat_file(sc_path, key='triangle')
                            fc_mat = load_mat_file(fc_path, key='fun_clean_cor')
                            sc_mats = [sc_mat]
                        else:
                            continue
                    else:
                        sc_mats, missing_sc = [], False
                        for sc_name in sc_netnames:
                            sc_base = osp.join(netDir, atlas_name, subj_for_file, "SC", sc_name)
                            sc_path = _resolve_conn_file(sc_base)
                            if sc_path is None:
                                missing_sc = True; break
                            sc_mats.append(load_connectivity_matrix(sc_path, isheader=True))
                        fc_base = osp.join(netDir, atlas_name, subj_for_file, "FC", fc_netname)
                        fc_path = _resolve_conn_file(fc_base)
                        if missing_sc or fc_path is None: continue
                        fc_mat = load_connectivity_matrix(fc_path, isheader=True)
                else:
                    matname = '{}.csv'.format(subj_for_file)
                    sc_mat = load_connectivity_matrix(osp.join(netDir, sc_netnames[0], matname))
                    sc_mats = [sc_mat]
                    fc_mat = load_connectivity_matrix(osp.join(netDir, fc_netname, matname), isheader=Isheader)

                if fc_mat.shape[0] != fc_mat.shape[1]:
                    continue
                n = fc_mat.shape[0]
                if any(sm.shape != (n, n) for sm in sc_mats):
                    continue
                if raw_mats_list and n != raw_mats_list[0]['fc_mat'].shape[0]:
                    continue
                raw_mats_list.append({'subj': subj, 'sc_mats': sc_mats, 'fc_mat': fc_mat})

        # --- 阶段 2：计算组水平的变异系数阈值掩码 (p=0.75) ---
        if raw_mats_list:
            n_nodes = raw_mats_list[0]['fc_mat'].shape[0]
            num_sc = len(raw_mats_list[0]['sc_mats'])
            global_mask = np.zeros((n_nodes, n_nodes), dtype=bool)

            print(">>> Calculating group consistency threshold (p=0.75)...")
            # 处理所有 SC
            W_thr_sc_list = []
            global_sc_max = []  # 【新增】：记录每种 SC 的全局最大值
            for sc_idx in range(num_sc):
                Ws_sc = np.stack([rm['sc_mats'][sc_idx] for rm in raw_mats_list], axis=2)
                
                # 【修改】：使用 99% 分位数代替绝对最大值，防止极端伪影碾压全局权重
                safe_max = np.percentile(np.abs(Ws_sc), 99.0)
                global_sc_max.append(safe_max if safe_max > 0 else 1.0)
                
                W_thr_sc = threshold_consistency(Ws_sc, 0.75)
                global_mask |= (W_thr_sc != 0)
                W_thr_sc_list.append(W_thr_sc)

            # 处理 FC
            Ws_fc = np.stack([rm['fc_mat'] for rm in raw_mats_list], axis=2)
            W_thr_fc = threshold_consistency(Ws_fc, 0.75)
            global_mask |= (W_thr_fc != 0)

            # --- 保存阈值处理相关数据到实验结果目录 ---
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                np.save(osp.join(output_dir, "thr_global_mask.npy"), global_mask.astype(np.uint8))
                np.save(osp.join(output_dir, "thr_fc.npy"), W_thr_fc)
                for sc_idx, W_thr_sc in enumerate(W_thr_sc_list):
                    np.save(osp.join(output_dir, f"thr_sc_{sc_idx}.npy"), W_thr_sc)
                print(f">>> Thresholding data saved to {output_dir}")

            # --- 阶段 3：使用阈值掩码构建稀疏的 PyG Graph ---
            print(">>> Building sparse PyG Data objects with Thresholding Mask...")
            for rm in raw_mats_list:
                subj, sc_mats, fc_mat = rm['subj'], rm['sc_mats'], rm['fc_mat']
                feat = torch.tensor(get_node_feature(n_nodes))
                edge_in, edge_out, edge_attr_l = [], [], []

                for i in range(n_nodes):
                    for j in range(i, n_nodes):
                        # 【核心修改】只保留一致性阈值过滤后的边（自环永远保留）
                        if global_mask[i, j] or global_mask[j, i] or i == j:
                            edge_in.append(i)
                            edge_out.append(j)
                            edge_attr = []
                            for sc_mat_item in sc_mats:
                                edge_attr.append(sc_mat_item[i][j] if i != j else 0)
                            edge_attr.append(fc_mat[i][j] if fc_mat[i][j]>0 else 0)
                            edge_attr.append(abs(fc_mat[i][j]) if fc_mat[i][j]<0 else 0)
                            edge_attr_l.append(edge_attr)

                edge_attr = torch.tensor(edge_attr_l, dtype=feat.dtype, device=feat.device)

                # ===================== 【新代码】全局归一化 + Fisher Z 变换 =====================
                for i in range(edge_attr.size(1)):
                    if i < num_sc:
                        # 【修改】：使用 99% 分位数归一化，并加入 clamp 防止极端值溢出
                        edge_attr[:, i] = torch.clamp(edge_attr[:, i] / global_sc_max[i], max=1.0)
                    else:
                        # 对于 FC 特征（原代码已分离为正相关和负相关的绝对值，区间为 [0, 1]）
                        # 使用 Fisher Z 变换展开分布: 0.5 * ln((1+r)/(1-r))
                        # 截断 r 在 0.99 以内，防止 log(0) 或产生无穷大(Inf)
                        fc_vals = torch.clamp(edge_attr[:, i], min=0.0, max=0.99)
                        edge_attr[:, i] = 0.5 * torch.log((1 + fc_vals) / (1 - fc_vals))
                # =============================================================================

                g_data = D.Data()
                g_data.x, g_data.edge_index, g_data.edge_attr = feat, torch.tensor([edge_in, edge_out]), edge_attr
                graph_dict[subj] = g_data
                new_graphs_processed = True

    # --- 阶段 4：为 Data 挂载当前的 Label 和 协变量 ---
    for k in range(dt.shape[0]):
        subj = str(dt[subject_col][k])
        if subj in subjlist_set and subj in graph_dict:
            raw_lb = lb_map[k]
            if pd.isna(raw_lb):
                skipped_bad_label += 1
                continue
            try:
                flb = float(raw_lb)
            except (TypeError, ValueError):
                skipped_bad_label += 1
                continue
            if not np.isfinite(flb):
                skipped_bad_label += 1
                continue

            g_data = graph_dict[subj]
            g_data.age = torch.tensor([age_map.get(subj, 0.0)], dtype=g_data.x.dtype)
            g_data.gender = torch.tensor([gender_map.get(subj, 0.0)], dtype=g_data.x.dtype)

            lb = torch.tensor(flb, dtype=g_data.x.dtype, device=g_data.x.device)
            if ifBucket:
                graph_list.append((g_data, torch.Tensor(label_bucketization(lb))))
            else:
                graph_list.append((g_data, lb))
    # ==================== 两阶段逻辑结束 ====================

    # ==================== 【核心新增 3：将新生成的图字典落盘】 ====================
    if new_graphs_processed:
        print(f">>> Saving processed graph caches to {cache_path} ...")
        torch.save(graph_dict, cache_path)
    # =========================================================================

    if skipped_bad_label:
        print(
            f">>> Skipped {skipped_bad_label} subjects with missing or non-finite "
            f"label ({labeltype!r})."
        )
    print("dataset size: {} subjects".format(len(graph_list)))
    if len(graph_list) == 0:
        hint = (
            f"No valid graphs under netDir={netDir!r}, atlas={atlas_name!r}. "
            "Check SC/FC CSVs exist, are non-empty, and names match dataset_cfg "
            f"(FC file key -> {fc_netname!r})."
        )
        raise RuntimeError(hint)
    return graph_list
