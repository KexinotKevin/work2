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
