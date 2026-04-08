import numpy as np
import torch
import nibabel
from nilearn.datasets import fetch_atlas_aal
from nilearn.plotting import find_xyz_cut_coords, plot_connectome

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_coords(atlas_img):
    # atlas_img = atlas['maps']
    # atlas_img = nibabel.load(atlas_img)
    # all ROIs except the background
    values = np.unique(atlas_img.get_fdata())[1:]
    # iterate over Harvard-Oxford ROIs
    coords = []
    for v in values:
        data = np.zeros_like(atlas_img.get_fdata())
        data[atlas_img.get_fdata() == v] = 1
        xyz = find_xyz_cut_coords(nibabel.Nifti1Image(data, atlas_img.affine))
        coords.append(xyz)
    return coords

def select_sig_attr_per_edge(edge_index, edge_attr):
    """return idx of significant edge attribute"""
    idmap = np.zeros((90, 90))
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        _, max_loc = torch.max(edge_attr[k,:], 0)
        idmap[i, j] = max_loc
        idmap[j, i] = max_loc
    return idmap


def make_sym_connectome(edge_index, edge_attr):
    scmat = np.zeros((90, 90))
    fcmat = np.zeros((90, 90))
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        scmat[i, j] = edge_attr[k, :][0]
        scmat[j, i] = edge_attr[k, :][0]
        if edge_attr[k, :][1] !=0:
            fcmat[i, j] = edge_attr[k, :][1]
            fcmat[j, i] = edge_attr[k, :][1]
        if edge_attr[k, :][2] !=0:
            fcmat[i, j] = edge_attr[k, :][2]
            fcmat[j, i] = edge_attr[k, :][2]
    return scmat, fcmat

# def plot_distribution(idmap):
#     subj_nums = idmap.shape[0]
#     row = idmap.shape[1]
#     col = idmap.shape[2]

#     # 统计每种取值在 [row, col] 位置上的频率
#     frequency = np.zeros((row, col, subj_nums))
#     for i in range(subj_nums):
#         frequency[:, :, i] = (idmap[i, :, :] == i).sum(axis=0)

#     # 将频率归一化
#     frequency_normalized = frequency / subj_nums

#     # 绘制二维热力图
#     plt.figure(figsize=(10, 10))
#     for i in range(subj_nums):
#         plt.subplot(1, subj_nums, i + 1)
#         sns.heatmap(frequency_normalized[:, :, i], annot=True, cmap="RdBu")
#         plt.title(f"Value {i} Frequency")

#     # 使用 t-test 检验显著性水平
#     p_values = np.ones((row, col, subj_nums, subj_nums))
#     for i in range(subj_nums):
#         for j in range(i + 1, subj_nums):
#             # 对每一对取值进行 t-test
#             t_stat, p_value = stats.ttest_ind(idmap[i, :, :], idmap[j, :, :], equal_var=False)
#             p_values[:, :, i, j] = p_value.reshape(row, col)
#             p_values[:, :, j, i] = p_value.reshape(row, col)  # 对称赋值

#     # 将 p_values 中的显著性水平应用到热力图上
#     for i in range(subj_nums):
#         for j in range(i + 1, subj_nums):
#             # 找到显著性小于 0.05 的位置
#             significant_positions = p_values[:, :, i, j] < 0.05
#             frequency_normalized[significant_positions, i] = np.nan
#             frequency_normalized[significant_positions, j] = np.nan

#     return plt, frequency_normalized

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

def plot_distribution(idmap, kind):
    subj_nums = idmap.shape[0]
    row = idmap.shape[1]
    col = idmap.shape[2]

    # 统计每种取值在 [row, col] 位置上的频率
    frequency = np.zeros((row, col, 3))  # 3种取值
    for i in range(3):
        frequency[:, :, i] = (idmap == i).sum(axis=0)  # 统计取值为1, 2, 3的频率

    # 将频率归一化
    frequency_normalized = frequency / subj_nums

    # 使用 t-test 检验显著性水平
    p_values = np.ones((row, col, 3, 3))  # 3种取值之间的显著性
    for i in range(3):
        for j in range(i + 1, 3):
            # 对每一对取值进行 t-test
            t_stat, p_value = stats.ttest_ind(idmap == (i + 1), idmap == (j + 1), axis=0)
            p_values[:, :, i, j] = p_value
            p_values[:, :, j, i] = p_value  # 对称赋值
    p_values[p_values == 0] = np.nan
    for i in range(3):
        for j in range(i+1, 3):
            significant_positions = p_values[:, :, i, j] < 0.05
            # frequency_normalized[significant_positions, i] = np.nan
            # frequency_normalized[significant_positions, j] = np.nan
            frequency[significant_positions, i] = np.nan
            frequency[significant_positions, j] = np.nan

            plt.figure(figsize=(20, 20))
            sns.heatmap(frequency[:, :, i], annot=False, cmap="RdBu_r", vmin=0)
            plt.title(f"Value {i + 1}-{j+1} Frequency (Significant)")
            plt.tight_layout()
            plt.savefig("significant_{}_{}_{}.pdf".format(kind, i+1, j+1), dpi = 800)


def build_saliency_matrix(edge_index, saliency_attr, num_nodes=None):
    """将一维的显著性向量还原为全连接矩阵 (动态推断节点数量)"""
    # 动态推断节点数：最大索引值 + 1
    if num_nodes is None:
        num_nodes = int(np.max(edge_index)) + 1
        
    sal_mat = np.zeros((num_nodes, num_nodes))
    
    # 将多关系的梯度求平均，代表这条边的整体重要性
    if len(saliency_attr.shape) > 1:
        importance = np.mean(saliency_attr, axis=1) 
    else:
        importance = saliency_attr

    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        sal_mat[i, j] = importance[k]
        sal_mat[j, i] = importance[k] # 保证对称
        
    return sal_mat


def plot_saliency_heatmaps(saliency_matrices, out_dir, label_name, sc_kinds=None, fc_kind=None):
    """绘制所有关系维度的显著性热力图，合并到一个PDF中
    
    布局：2x2网格
    第一行：relation 0 (FA) 和 relation 1 (fiber_count)
    第二行：relation 2 (fc_kind_pos) 和 relation 3 (fc_kind_neg)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    mean_saliency = np.mean(saliency_matrices, axis=0)
    num_relations = mean_saliency.shape[-1] if len(mean_saliency.shape) == 3 else 1
    
    if num_relations < 3:
        raise ValueError(f"Expected at least 3 relations, got {num_relations}")
    
    if sc_kinds is None:
        sc_kinds = ['fiber_count']
    if fc_kind is None:
        fc_kind = 'pcc_rest'
    
    # 根据 num_relations 动态设置关系名称和布局
    if num_relations == 3:
        rel_names = [sc_kinds[0] if len(sc_kinds) > 0 else f'Relation_0',
                     sc_kinds[1] if len(sc_kinds) > 1 else f'Relation_1',
                     f'{fc_kind}_pos']
        rows, cols = 1, 3
    else:  # num_relations >= 4
        rel_names = [sc_kinds[0] if len(sc_kinds) > 0 else f'Relation_0',
                     sc_kinds[1] if len(sc_kinds) > 1 else f'Relation_1',
                     f'{fc_kind}_pos',
                     f'{fc_kind}_neg']
        rows, cols = 2, 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    fig.suptitle(f'Saliency Heatmaps: {label_name}', fontsize=16, fontweight='bold', y=1.02)
    
    for r in range(num_relations):
        rel_saliency = mean_saliency[:, :, r] if num_relations > 1 else mean_saliency
        rel_name = rel_names[r] if r < len(rel_names) else f'Relation_{r}'
        # plt.subplots(1, n) returns a 1D axes array; (n, m) returns 2D — ravel matches row-major r
        ax = np.ravel(axes)[r]
        non_zero_mask = (rel_saliency != 0)
        if non_zero_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{rel_name}\n(All zeros)', ha='center', va='center', fontsize=12)
        else:
            valid_data = rel_saliency[non_zero_mask]
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
            if vmin == vmax:
                vmin, vmax = 0, 1
            
            sns.heatmap(rel_saliency, cmap="Reds", square=True, mask=(rel_saliency == 0), 
                        vmin=vmin, vmax=vmax, ax=ax, cbar=True, cbar_kws={'shrink': 0.6})
        ax.set_title(f'{rel_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    true_out_dir = os.path.join(out_dir, label_name)
    os.makedirs(true_out_dir, exist_ok=True)
    pdf_path = os.path.join(true_out_dir, f'saliency_heatmap_{label_name}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmaps to {pdf_path}")


def _get_cortical_mask(coords, threshold=None):
    """根据Y坐标自动识别皮层节点
    
    BNA246等脑模板中，皮层节点通常具有较高的Y值。
    自动检测方法：使用Y值的60%分位数作为阈值，可适应不同脑图谱。
    """
    if coords is None or len(coords) == 0:
        return None
    
    y_values = coords[:, 1]
    
    if threshold is None:
        # 自动检测：使用60%分位数作为皮层阈值
        threshold = np.percentile(y_values, 60)
    
    return y_values >= threshold

def _normalize_by_region(saliency_matrix, cortical_mask):
    """对皮层和非皮层区域的边分别做min-max归一化，再合并"""
    import numpy as np
    
    result = np.zeros_like(saliency_matrix)
    n = saliency_matrix.shape[0]
    cortical_set = set(np.where(cortical_mask)[0]) if cortical_mask is not None else set()
    
    # 收集皮层边和非皮层边的值
    cortical_edges = []
    non_cortical_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            val = saliency_matrix[i, j]
            if val > 0:
                if cortical_mask is not None and i in cortical_set and j in cortical_set:
                    cortical_edges.append(val)
                else:
                    non_cortical_edges.append(val)
    
    # 分别归一化
    def minmax_norm(values):
        if len(values) == 0:
            return lambda x: 0
        vmin, vmax = np.min(values), np.max(values)
        if vmax > vmin:
            return lambda x: (x - vmin) / (vmax - vmin)
        return lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0
    
    cortical_norm = minmax_norm(cortical_edges)
    non_cortical_norm = minmax_norm(non_cortical_edges)
    
    # 填充结果矩阵（对称）
    for i in range(n):
        for j in range(i + 1, n):
            val = saliency_matrix[i, j]
            if val > 0:
                if cortical_mask is not None and i in cortical_set and j in cortical_set:
                    norm_val = cortical_norm(val)
                else:
                    norm_val = non_cortical_norm(val)
                result[i, j] = norm_val
                result[j, i] = norm_val
    
    return result


def plot_saliency_connectomes(saliency_matrices, coords, out_dir, label_name, sc_kinds=None, fc_kind=None):
    """绘制各关系维度的大脑连接图，每个relation单独保存一个PDF
    
    命名规则：
    - relation 0: FA
    - relation 1: fiber_count
    - relation 2: fc_kind_pos
    - relation 3: fc_kind_neg
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_connectome

    mean_saliency = np.mean(saliency_matrices, axis=0)
    num_relations = mean_saliency.shape[-1] if len(mean_saliency.shape) == 3 else 1
    
    if num_relations < 3:
        raise ValueError(f"Expected at least 3 relations, got {num_relations}")
    
    if sc_kinds is None:
        sc_kinds = ['FA', 'fiber_count']
    if fc_kind is None:
        fc_kind = 'pcc_rest'
    
    # 根据 num_relations 动态设置关系名称
    if num_relations == 3:
        rel_names = [sc_kinds[0] if len(sc_kinds) > 0 else f'Relation_0',
                     sc_kinds[1] if len(sc_kinds) > 1 else f'Relation_1',
                     f'{fc_kind}_pos']
    else:  # num_relations >= 4
        rel_names = [sc_kinds[0] if len(sc_kinds) > 0 else f'Relation_0',
                     sc_kinds[1] if len(sc_kinds) > 1 else f'Relation_1',
                     f'{fc_kind}_pos',
                     f'{fc_kind}_neg']
    
    true_out_dir = os.path.join(out_dir, label_name)
    os.makedirs(true_out_dir, exist_ok=True)
    
    # 识别皮层节点（自动检测阈值）
    cortical_mask = _get_cortical_mask(coords)
    
    for r in range(num_relations):
        rel_saliency = mean_saliency[:, :, r] if num_relations > 1 else mean_saliency
        rel_name = rel_names[r] if r < len(rel_names) else f'Relation_{r}'
        
        # 【核心修改】对皮层和非皮层区域分别归一化
        if cortical_mask is not None:
            rel_saliency = _normalize_by_region(rel_saliency, cortical_mask)
        
        non_zero_saliency = rel_saliency[rel_saliency > 0]
        
        if len(non_zero_saliency) > 0 and len(coords) == rel_saliency.shape[0]:
            threshold = np.percentile(non_zero_saliency, 90)
            
            fig = plt.figure(figsize=(10, 5))
            plot_connectome(rel_saliency, coords, edge_threshold=threshold, 
                           title=f"Top Saliency Connectome: {label_name} ({rel_name})",
                           figure=fig, node_size=20, edge_kwargs={'lw': 2})
            
            pdf_path = os.path.join(true_out_dir, f'saliency_connectome_{label_name}_{rel_name}.pdf')
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved connectome to {pdf_path}")
        else:
            print(f"Warning: Skipping connectome for {rel_name} - no valid data or coordinate mismatch.")
