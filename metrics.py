import torch
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