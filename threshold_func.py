import numpy as np

def threshold_arbmeasure(W, M, p):
    """
    根據給定的任意測量指標 M 進行閾值過濾，保留排序最高的邊。
    
    參數:
        W (ndarray): N x N 加權或二值連接矩陣
        M (ndarray): N x N 測量指標矩陣，用來排序 W 中的邊
        p (float): 要保留的權重比例 (0 < p <= 1)
        
    返回:
        W_thr (ndarray): 閾值過濾後的連接矩陣
    """
    W_thr = W.copy()
    n = W_thr.shape[0]
    
    # 清除對角線 (自我連接設為 0)
    np.fill_diagonal(W_thr, 0)
    
    # 檢查是否為對稱矩陣 (使用 np.allclose 允許微小的浮點數誤差)
    is_symmetric = np.allclose(W_thr, W_thr.T, equal_nan=True)
    
    if is_symmetric:
        # 如果是對稱矩陣，只取上三角進行處理，避免重複計算
        W_work = np.triu(W_thr)
        ud = 2  # 將需要移除的邊數減半
    else:
        W_work = W_thr
        ud = 1
        
    # 找出所有非零的連接邊的索引
    ind_r, ind_c = np.nonzero(W_work)
    
    # 根據測量指標 M 的值進行排序 (降序)
    m_vals = M[ind_r, ind_c]
    # np.argsort 預設是升序，加上負號變為降序
    sorted_indices = np.argsort(-m_vals) 
    
    # 計算需要保留的邊的數量
    # 總可能連接數為 n^2 - n (不含對角線)
    en = int(np.round((n**2 - n) * p / ud))
    
    # 找出需要被歸零（被閾值過濾掉）的邊的索引
    remove_idx = sorted_indices[en:]
    
    if is_symmetric:
        # 如果是對稱矩陣，先將 W_thr 變為純上三角矩陣
        W_thr = np.triu(W_thr)
        
    # 應用閾值，將排名靠後的邊設為 0
    if len(remove_idx) > 0:
        W_thr[ind_r[remove_idx], ind_c[remove_idx]] = 0
        
    if is_symmetric:
        # 重建對稱矩陣
        W_thr = W_thr + W_thr.T
        
    return W_thr

def threshold_consistency(Ws, p):
    """
    基於一致性對網絡邊緣進行閾值處理。
    
    參數:
        Ws (ndarray): N x N x M 的 3D 陣列，包含 M 個受試者的連接矩陣
        p (float): 要保留的權重比例 (0 < p <= 1)
        
    返回:
        W_thr (ndarray): 閾值過濾後的群體平均連接矩陣
    """
    # 沿著第三個維度（受試者維度）計算平均值
    Wmean = np.mean(Ws, axis=2)
    
    # 計算標準差 (ddof=0 確保行為與 MATLAB 的 std(..., 0, 3) 一致)
    Wstd = np.std(Ws, axis=2, ddof=0)
    
    # 計算變異係數 (Coefficient of variation: 標差 / 平均值)
    # 使用 np.errstate 避免除以 0 時拋出警告，並將無效值填為 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Wcv = np.where(Wmean != 0, Wstd / Wmean, 0)
        
    # 以負的變異係數（-Wcv）作為測量指標傳入，
    # 這樣 threshold_arbmeasure 保留最大值時，實際上就是保留 CV 最小值（一致性最高）
    W_thr = threshold_arbmeasure(Wmean, -Wcv, p)
    
    return W_thr

if __name__ == "__main__":
    # 隨機生成模擬數據 (90個節點, 10個受試者)
    # Ws.shape 應為 (90, 90, 10)
    np.random.seed(42)
    Ws_mock = np.random.rand(90, 90, 10) 

    # 執行一致性閾值過濾，保留 20% (p=0.75)（变异系数在第75百分位）
    W_thresholded = threshold_consistency(Ws_mock, 0.75)
    print(W_thresholded)