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


def plot_saliency_results(saliency_matrices, coords, out_dir, label_name):
    """绘制各关系维度的平均显著性热力图和Top-K大脑连接图"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from nilearn.plotting import plot_connectome

    # saliency_matrices 现在的 shape 是 (samples, num_nodes, num_nodes, num_relations)
    mean_saliency = np.mean(saliency_matrices, axis=0) 
    
    # 兼容单关系和多关系
    num_relations = mean_saliency.shape[-1] if len(mean_saliency.shape) == 3 else 1
    
    # 你可以根据实际情况修改这里的别名，比如 ['SC', 'FC_Rest', 'FC_Task']
    rel_names = [f"Relation_{r}" for r in range(num_relations)] 

    for r in range(num_relations):
        rel_saliency = mean_saliency[:, :, r] if num_relations > 1 else mean_saliency
        rel_name = rel_names[r]
        
        # 1. 绘制特定关系维度的显著性热力图 (Heatmap)
        plt.figure(figsize=(10, 8))
        sns.heatmap(rel_saliency, cmap="Reds", square=True)
        plt.title(f"Average Saliency Map: {label_name} ({rel_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"saliency_heatmap_{label_name}_{rel_name}.pdf"), dpi=300)
        plt.close()

        # 2. 绘制特定关系维度的大脑 3D 连接图 (Connectome)
        non_zero_saliency = rel_saliency[rel_saliency > 0]
        if len(non_zero_saliency) > 0:
            threshold = np.percentile(non_zero_saliency, 98)
            
            if len(coords) == rel_saliency.shape[0]:
                fig = plt.figure(figsize=(10, 5))
                plot_connectome(rel_saliency, coords, edge_threshold=threshold, 
                                title=f"Top Saliency Connectome: {label_name} ({rel_name})",
                                figure=fig, node_size=20, edge_kwargs={'lw': 2})
                fig.savefig(os.path.join(out_dir, f"saliency_connectome_{label_name}_{rel_name}.pdf"), dpi=300)
                plt.close()
            else:
                print(f"Warning: Connectome plot skipped for {label_name} ({rel_name}). Coordinates mismatch.")
