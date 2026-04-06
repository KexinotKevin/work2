import torch
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

def r_squared(y_true, y_pred):
    """
    计算R方（R-squared）。
    :param y_true: 真实标签，形状为 (n_samples,)
    :param y_pred: 预测标签，形状为 (n_samples,)
    :return: R方值
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def rmse(y_true, y_pred):
    """
    计算均方根误差（RMSE）。
    :param y_true: 真实标签，形状为 (n_samples,)
    :param y_pred: 预测标签，形状为 (n_samples,)
    :return: RMSE值
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def mae(y_true, y_pred):
    """
    计算平均绝对误差（MAE）。
    :param y_true: 真实标签，形状为 (n_samples,)
    :param y_pred: 预测标签，形状为 (n_samples,)
    :return: MAE值
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()


def concordance_correlation_coefficient(y_true, y_pred):
    """
    计算一致性相关系数（CCC, Concordance Correlation Coefficient）。
    CCC = 2 * cov(y_true, y_pred) / (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
    :param y_true: 真实标签，NumPy array
    :param y_pred: 预测标签，NumPy array
    :return: CCC值
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    
    # 计算协方差
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    
    numerator = 2 * cov
    denominator = var_t + var_p + (mean_t - mean_p) ** 2
    
    if denominator == 0:
        return np.nan
        
    return numerator / denominator