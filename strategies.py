"""
训练策略模块：包含早停机制和动态学习率调度器
"""

import numpy as np


class EarlyStopping:
    """早停机制，当验证集性能在若干轮内没有改善时停止训练
    
    策略：
    1. 监控验证集 loss（默认）
    2. 支持恢复最佳模型
    3. 可设置最小改善阈值，避免因噪声而误判
    4. 支持设置最小训练轮数，防止过早停止
    """
    
    def __init__(self, patience=15, min_delta=1e-4, mode='min', 
                 restore_best_weights=True, min_epochs=10, verbose=True):
        """
        Args:
            patience: 容忍多少轮性能没有改善（默认15）
            min_delta: 被认为是改善的最小变化量（默认1e-4）
            mode: 'min' 表示监控最小值（如loss），'max' 表示监控最大值（如accuracy）
            restore_best_weights: 是否在停止时恢复最佳权重（默认True）
            min_epochs: 最小训练轮数，即使触发早停也要训练至少这么多轮（默认10）
            verbose: 是否打印早停相关信息（默认True）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
        self.best_model_state = None
        
        # 根据模式确定比较函数
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
            self.score_fn = lambda x: x
        else:
            self.is_better = lambda score, best: score > best + min_delta
            self.score_fn = lambda x: -x  # 对于max模式，取负值使得越小越好
    
    def __call__(self, epoch, score, model_state=None):
        """
        检查是否应该早停
        
        Args:
            epoch: 当前epoch（从0开始）
            score: 当前验证集分数（loss或accuracy）
            model_state: 当前模型状态字典（可选）
            
        Returns:
            bool: 是否应该停止训练
        """
        # 最小训练轮数检查
        if epoch < self.min_epochs:
            if self.best_score is None:
                self.best_score = self.score_fn(score)
                self.best_epoch = epoch
                if model_state is not None:
                    self.best_model_state = {k: v.cpu().clone() for k, v in model_state.items()}
            return False
        
        current_score = self.score_fn(score)
        
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if model_state is not None:
                self.best_model_state = {k: v.cpu().clone() for k, v in model_state.items()}
        elif self.is_better(score, self.best_score):
            # 性能改善
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if model_state is not None and self.restore_best_weights:
                self.best_model_state = {k: v.cpu().clone() for k, v in model_state.items()}
            if self.verbose:
                print(f"    [EarlyStopping] Validation improved! Resetting counter.")
        else:
            # 性能没有改善
            self.counter += 1
            if self.verbose:
                print(f"    [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"    [EarlyStopping] Early stopping triggered at epoch {epoch + 1}!")
                    print(f"    [EarlyStopping] Best epoch was {self.best_epoch + 1} with score {self.score_fn(current_score) if isinstance(current_score, (int, float)) else self.best_score:.6f}")
                return True
        
        return False
    
    def restore_weights(self, model):
        """恢复到最佳模型权重"""
        if self.best_model_state is not None and self.restore_best_weights:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"    [EarlyStopping] Restored best model weights from epoch {self.best_epoch + 1}")
            return True
        return False
    
    def get_best_epoch(self):
        """返回最佳epoch（从0开始）"""
        return self.best_epoch
    
    def get_wait_count(self):
        """返回当前连续未改善的轮数"""
        return self.counter


class DynamicLearningRateScheduler:
    """动态学习率调度器，结合loss plateau检测和梯度监控
    
    策略：
    1. 当loss plateau时降低学习率（patience轮内未改善）
    2. 当梯度范数持续过小时适当增大学习率
    3. 当梯度范数过大时减小学习率防止震荡
    4. 支持warmup阶段平滑起步
    """
    
    def __init__(self, optimizer, base_lr, min_lr=1e-6, 
                 patience=10, factor=0.5, warmup_epochs=5,
                 grad_norm_thresh_low=0.01, grad_norm_thresh_high=10.0,
                 increase_lr_factor=1.05, window_size=5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.warmup_epochs = warmup_epochs
        self.grad_norm_thresh_low = grad_norm_thresh_low
        self.grad_norm_thresh_high = grad_norm_thresh_high
        self.increase_lr_factor = increase_lr_factor
        self.window_size = window_size
        
        self.current_lr = base_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.grad_history = []
        self.loss_history = []
        self.increase_count = 0  # 防止无限增大学习率
        
    def step(self, epoch, loss, grad_norm=None):
        """更新学习率
        
        Args:
            epoch: 当前epoch
            loss: 当前epoch的loss
            grad_norm: 当前batch的梯度范数（可选）
        """
        self.loss_history.append(loss)
        
        # Warmup阶段：线性增加学习率
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            self._set_lr(lr)
            return lr
            
        # 记录梯度范数
        if grad_norm is not None:
            self.grad_history.append(grad_norm)
            if len(self.grad_history) > self.window_size:
                self.grad_history.pop(0)
        
        # 策略1: Loss plateau检测
        if loss < self.best_loss - 1e-6:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # 策略2: 基于梯度范数的调整
        lr_adjustment = 1.0
        if len(self.grad_history) >= self.window_size:
            avg_grad = np.mean(self.grad_history)
            
            # 梯度过小：可能陷入plateau，适当增大学习率
            if avg_grad < self.grad_norm_thresh_low and self.increase_count < 3:
                lr_adjustment = self.increase_lr_factor
                self.increase_count += 1
            # 梯度过大：可能震荡，减小学习率
            elif avg_grad > self.grad_norm_thresh_high:
                lr_adjustment = 0.7
                self.increase_count = 0
            else:
                self.increase_count = 0
        
        # 综合调整
        if self.wait_count >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait_count = 0
            self.increase_count = 0  # 重置增加计数
        else:
            new_lr = self.current_lr * lr_adjustment
            new_lr = max(new_lr, self.min_lr)
            new_lr = min(new_lr, self.base_lr)  # 不超过初始学习率
            
        if abs(new_lr - self.current_lr) > 1e-9:
            self._set_lr(new_lr)
            
        return self.current_lr
    
    def _set_lr(self, lr):
        """设置所有参数组的学习率"""
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        return self.current_lr
