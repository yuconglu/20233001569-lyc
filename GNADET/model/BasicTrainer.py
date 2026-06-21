import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import torch.nn.functional as F
import optuna # 为剪枝功能导入optuna

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, climatology_unnormalized, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        
        # 存储气候学数据
        self.climatology_unnormalized = torch.from_numpy(climatology_unnormalized).float().to(self.args.device)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        # 初始化Markdown日志文件
        self.markdown_log_file_path = "/root/autodl-tmp/AddPiData/BestGIS/NADE-main/model/训练记录.md"
        self.logger.info(f"Markdown log file path set to: {self.markdown_log_file_path}")
        markdown_log_dir = os.path.dirname(self.markdown_log_file_path)
        try:
            if not os.path.exists(markdown_log_dir):
                os.makedirs(markdown_log_dir, exist_ok=True)
                self.logger.info(f"Created markdown log directory: {markdown_log_dir}")
            else:
                self.logger.info(f"Markdown log directory already exists: {markdown_log_dir}")

            if not os.access(markdown_log_dir, os.W_OK):
                self.logger.error(f"Error: Markdown log directory is not writable: {markdown_log_dir}")
                self.markdown_log_file_path = None 
            elif os.path.exists(self.markdown_log_file_path) and not os.access(self.markdown_log_file_path, os.W_OK):
                 self.logger.error(f"Error: Markdown log file exists but is not writable: {self.markdown_log_file_path}")
                 self.markdown_log_file_path = None
            
            if self.markdown_log_file_path: 
                with open(self.markdown_log_file_path, 'a', encoding='utf-8') as f:
                    self.logger.info(f"Attempting to write initial session info to markdown log: {self.markdown_log_file_path}")
                    f.write(f"\n--- New Training Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    f.write(f"Log Directory for this session: {self.args.log_dir}\n")
                    f.write(f"Trainer args: {vars(self.args)}\n") # 记录参数以供调试
                self.logger.info(f"Successfully wrote initial session info to markdown log.")
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR during markdown log initialization: {e}", exc_info=True)
            self.markdown_log_file_path = None

        # 初始化用于存储最佳验证加权RMSE的变量
        self.best_avg_val_weighted_rmse = float('inf')
        # 新增：记录获得最优加权RMSE的epoch
        self.best_avg_val_weighted_rmse_epoch = -1  # -1表示尚未获得
        # 新增：用于记录最佳ACC的变量和对应的epoch
        self.best_avg_val_weighted_acc = float('-inf') # ACC越高越好
        self.best_acc_epoch = -1 # -1表示尚未获得
        # 新增：记录次佳指标
        self.second_best_avg_val_weighted_rmse = float('inf')
        self.second_best_avg_val_weighted_rmse_epoch = -1
        self.second_best_avg_val_weighted_acc = float('-inf')
        self.second_best_acc_epoch = -1

        # 加载纬度数据
        # 路径需要根据实际数据存储位置调整，这里假设与X_GIS.npy在同一目录，由dataloader.py中的base_data_path决定
        # 在 get_dataloader 中确定 base_data_path 的逻辑需要统一
        # 硬编码回退路径，与dataloader.py中的逻辑类似
        base_data_path = None

        # 0. 最优先使用 args.data_dir (如果通过 Run.py 从 GIS.conf 传入且有效)
        if hasattr(args, 'data_dir') and args.data_dir and os.path.isdir(args.data_dir):
            base_data_path = args.data_dir
            self.logger.info(f"Trainer: 使用来自 args.data_dir (源自GIS.conf) 的路径加载纬度数据: {base_data_path}")

        # 1. 其次，尝试使用 args.processed_data_dir (如果提供且有效，并且尚未通过 args.data_dir 设置)
        if base_data_path is None and hasattr(args, 'processed_data_dir') and args.processed_data_dir and os.path.isdir(args.processed_data_dir):
            base_data_path = args.processed_data_dir
            self.logger.info(f"Trainer: 使用提供的 processed_data_dir 加载纬度数据: {base_data_path}")

        # 2. 如果没有，尝试从 args.dataset 推断 (如果它是文件路径，并且尚未设置 base_data_path)
        elif base_data_path is None and os.path.isfile(args.dataset):
            base_data_path = os.path.dirname(args.dataset)
            self.logger.info(f"Trainer: 从 args.dataset 文件路径推断的目录加载纬度数据: {base_data_path}")
        
        # 3. 如果还是没有，并且 args.dataset 是一个存在的目录 (并且尚未设置 base_data_path)
        elif base_data_path is None and os.path.isdir(args.dataset):
            base_data_path = args.dataset # args.dataset 本身就是目录
            self.logger.info(f"Trainer: 使用 args.dataset 作为目录加载纬度数据: {base_data_path}")

        if base_data_path is None or not os.path.isdir(base_data_path):
            base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') # 回退到 NADE-main/data
            self.logger.warning(f"Trainer:无法从参数确定数据目录，回退到 {base_data_path} 以加载纬度数据")

        lat_coords_filename = f'lat_coords_{args.dataset}.npy' # 例如 lat_coords_GIS.npy
        lat_coords_path = os.path.join(base_data_path, lat_coords_filename)
        try:
            self.logger.info(f"加载纬度坐标文件: {lat_coords_path}")
            latitudes_np = np.load(lat_coords_path)
            self.latitudes = torch.from_numpy(latitudes_np).float().to(self.args.device) # 形状 (num_nodes,)
            if self.latitudes.shape[0] != self.args.num_nodes:
                self.logger.error(f"加载的纬度坐标数量 ({self.latitudes.shape[0]}) 与配置的节点数 ({self.args.num_nodes}) 不匹配！")
                raise ValueError("纬度坐标与节点数不匹配")
            self.logger.info(f"纬度坐标加载成功，形状: {self.latitudes.shape}")
        except FileNotFoundError:
            self.logger.error(f"错误: 纬度坐标文件 {lat_coords_path} 未找到。加权指标计算将无法进行。")
            self.latitudes = None # 设置为None，后续检查
        except Exception as e_lat_load:
            self.logger.error(f"加载或处理纬度坐标时出错: {e_lat_load}")
            self.latitudes = None

    # 辅助函数：计算纬度权重 (可以设为静态或实例方法)
    def get_latitude_weights(self, latitudes_tensor_1d):
        """ 计算并归一化纬度权重 """
        if latitudes_tensor_1d is None:
            self.logger.warning("纬度数据未加载，无法计算纬度权重。返回None。")
            # Markdown log - 虽然是warning，但为了完整性可以考虑是否记录
            # self._log_to_markdown("Warning: 纬度数据未加载，无法计算纬度权重。返回None。") 
            return None
        weights = torch.cos(torch.deg2rad(latitudes_tensor_1d))
        weights = weights / weights.mean() # 归一化
        return weights

    def _log_to_markdown(self, message_content):
        """将带有当前时间戳的消息内容追加到Markdown日志文件"""
        if not self.markdown_log_file_path: 
            return

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"{timestamp}: {message_content}" 
        try:
            with open(self.markdown_log_file_path, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write to markdown log file {self.markdown_log_file_path}: {e}", exc_info=True)
            # self.markdown_log_file_path = None # 可选：一旦失败就彻底禁用，以避免日志刷屏

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        y_pred_list = [] # 使用 y_pred_list 和 y_true_list 来避免早期版本的拼写错误
        y_true_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                y_true_list.append(label)
                y_pred_list.append(output)
        
        y_true_scaled = torch.cat(y_true_list, dim=0) # 标准化后的真实值 (B, H, N, D_out)
        y_pred_scaled = torch.cat(y_pred_list, dim=0) # 标准化后的预测值 (B, H, N, D_out)

        # 首先计算原始损失 (基于标准化或逆标准化的值，取决于self.loss的定义)
        # 确保y_pred和y_true在计算loss前具有相同的缩放状态
        # 如果 self.loss 内部处理逆标准化，则传入 scaled values
        # 如果 self.loss 期望真实值，则需要在此处逆标准化
        # 假设: self.loss 函数期望的 preds 和 labels 都已经是被 self.scaler.inverse_transform 处理过的值（如果real_value=False）
        # 或者，如果real_value=True, 期望的是模型直接输出的、未经过inverse_transform的值（但label通常是inverse_transform过的）

        # 为了与现有loss计算逻辑兼容，我们先按原样计算 val_loss
        # 然后，为了指标计算，我们确保 y_true 和 y_pred 都是逆标准化后的真实值
        y_true = self.scaler.inverse_transform(y_true_scaled)[:,:,:,:self.args.output_dim] # (B, H, N, D_out)
        if self.args.real_value:
            # 如果real_value为True, 模型输出已经是真实尺度，不需要scaler.inverse_transform
            # 但loss计算时，label端通常是逆变换过的。这里我们假设y_pred_scaled已经是真实尺度
            y_pred = y_pred_scaled[:,:,:,:self.args.output_dim] 
        else:
            y_pred = self.scaler.inverse_transform(y_pred_scaled)[:,:,:,:self.args.output_dim]
        
        # 重新计算 val_loss 以确保使用的是与下面指标计算一致的y_pred和y_true
        # 这取决于 self.loss 是否期望 cuda 张量，以及它们是否已经是逆标准化的
        # 如果 self.loss 是例如 nn.MSELoss() 并且 self.args.real_value=False,
        # 它通常作用于模型输出 (y_pred_scaled) 和目标 (y_true_scaled) 上。
        # 如果 self.args.real_value=True，loss作用于 y_pred_scaled 和 y_true (逆标准化)。
        # 为了演示，我们将坚持使用原始的val_loss计算，并仅为新指标使用完全逆标准化的值。
        # 或者，我们可以调整val_loss的计算：
        if self.args.real_value:
             val_loss_recalc = self.loss(y_pred_scaled.cuda(), y_true.cuda()) # y_pred_scaled是模型输出，y_true是逆变换的
        else:
             val_loss_recalc = self.loss(y_pred.cuda(), y_true.cuda()) # 两者都是逆变换的
        # self.logger.info(f'Debug: Original val_loss: {original_val_loss.item()}, Recalculated val_loss: {val_loss_recalc.item()}')
        # 替换为原始的val_loss计算，以减少不必要的改动
        original_val_loss = self.loss(y_pred_scaled.cuda() if self.args.real_value else y_pred.cuda(), 
                                    y_true_scaled.cuda() if not self.args.real_value else y_true.cuda())

        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, original_val_loss.item()))
        log_content_val_loss = '**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, original_val_loss.item())
        self._log_to_markdown(log_content_val_loss)

        # --- 新增：计算加权 RMSE 和 ACC --- (在CPU上进行，以匹配原始MAE/RMSE逻辑)
        y_pred_metrics = y_pred.cpu() # (B, H, N, D_out)
        y_true_metrics = y_true.cpu() # (B, H, N, D_out)
        
        avg_val_rmse_w = float('inf') # 初始化本轮验证的加权RMSE

        if self.latitudes is not None:
            latitude_weights = self.get_latitude_weights(self.latitudes.cpu()) # (N,)
            # 气候学数据 (N, D_out) -> (N,) (取第一个输出维度)
            clim_for_acc = self.climatology_unnormalized.cpu()[:, :1].squeeze(-1) # (N,)
            if latitude_weights is None:
                self.logger.warning("无法获取纬度权重，跳过加权指标计算。")
                return original_val_loss.item() # 或者返回包含RMSE的元组/字典

            val_rmse_weighted_list = []
            val_acc_weighted_list = []

            num_horizon_steps = y_pred_metrics.shape[1]
            for h_idx in range(num_horizon_steps):
                y_pred_h = y_pred_metrics[:, h_idx, :, 0] # (B, N), 假设D_out=1或只关心第一个
                y_true_h = y_true_metrics[:, h_idx, :, 0] # (B, N)

                # 计算加权 RMSE
                error_h = y_pred_h - y_true_h # (B, N)
                # 权重 (N,) -> (1, N) for broadcasting
                weighted_squared_error_h = (error_h**2) * latitude_weights.unsqueeze(0) # (B,N) * (1,N) -> (B,N)
                # 对所有样本和节点求平均
                current_rmse_w = torch.sqrt(weighted_squared_error_h.mean()) 
                val_rmse_weighted_list.append(current_rmse_w.item())
                self.logger.info('  Val Horizon {:02d}: Weighted RMSE: {:.6f}'.format(h_idx + 1, current_rmse_w.item()))
                log_content_h_rmse = '  Val Horizon {:02d}: Weighted RMSE: {:.6f}'.format(h_idx + 1, current_rmse_w.item())
                self._log_to_markdown(log_content_h_rmse)

                # 计算加权 ACC
                # 距平 anom: (B, N)
                pred_anom_h = y_pred_h - clim_for_acc.unsqueeze(0) # (B,N) - (1,N) -> (B,N)
                true_anom_h = y_true_h - clim_for_acc.unsqueeze(0) # (B,N) - (1,N) -> (B,N)
                
                # 减去整个 (B,N) 张量的均值得到 prime
                pred_anom_prime_h = pred_anom_h - pred_anom_h.mean() # (B, N)
                true_anom_prime_h = true_anom_h - true_anom_h.mean() # (B, N)
                
                # (B,N) * (B,N) * (1,N) element-wise, then sum over B and N
                numerator = (pred_anom_prime_h * true_anom_prime_h * latitude_weights.unsqueeze(0)).sum()
                denominator_pred_sq = ((pred_anom_prime_h**2) * latitude_weights.unsqueeze(0)).sum()
                denominator_true_sq = ((true_anom_prime_h**2) * latitude_weights.unsqueeze(0)).sum()
                
                current_acc_w = numerator / (torch.sqrt(denominator_pred_sq * denominator_true_sq) + 1e-6)
                val_acc_weighted_list.append(current_acc_w.item())
                self.logger.info('  Val Horizon {:02d}: Weighted ACC:  {:.6f}'.format(h_idx + 1, current_acc_w.item()))
                log_content_h_acc = '  Val Horizon {:02d}: Weighted ACC:  {:.6f}'.format(h_idx + 1, current_acc_w.item())
                self._log_to_markdown(log_content_h_acc)
            
            if val_rmse_weighted_list: # 确保列表非空
                avg_val_rmse_w = np.mean(val_rmse_weighted_list)
                self.logger.info('  Validation Avg Weighted RMSE: {:.6f}'.format(avg_val_rmse_w))
                log_content_avg_rmse = '  Validation Avg Weighted RMSE: {:.6f}'.format(avg_val_rmse_w)
                self._log_to_markdown(log_content_avg_rmse)
                # 更新 Trainer 实例中记录的最佳（最小）avg_val_rmse_w
                if avg_val_rmse_w < self.best_avg_val_weighted_rmse:
                    # 原来的最佳变成次佳
                    self.second_best_avg_val_weighted_rmse = self.best_avg_val_weighted_rmse
                    self.second_best_avg_val_weighted_rmse_epoch = self.best_avg_val_weighted_rmse_epoch
                    # 新的变成最佳
                    self.best_avg_val_weighted_rmse = avg_val_rmse_w
                    self.best_avg_val_weighted_rmse_epoch = epoch  # 记录获得最优的epoch
                elif avg_val_rmse_w < self.second_best_avg_val_weighted_rmse and avg_val_rmse_w != self.best_avg_val_weighted_rmse:
                    # 如果新值介于最佳和次佳之间 (且不等于最佳，以防重复记录)
                    self.second_best_avg_val_weighted_rmse = avg_val_rmse_w
                    self.second_best_avg_val_weighted_rmse_epoch = epoch
            if val_acc_weighted_list:
                avg_val_acc_w = np.mean(val_acc_weighted_list)
                self.logger.info('  Validation Avg Weighted ACC:  {:.6f}'.format(avg_val_acc_w))
                log_content_avg_acc = '  Validation Avg Weighted ACC:  {:.6f}'.format(avg_val_acc_w)
                self._log_to_markdown(log_content_avg_acc)
                # 新增：更新最佳ACC和其对应的epoch
                if avg_val_acc_w > self.best_avg_val_weighted_acc:
                    # 原来的最佳变成次佳
                    self.second_best_avg_val_weighted_acc = self.best_avg_val_weighted_acc
                    self.second_best_acc_epoch = self.best_acc_epoch
                    # 新的变成最佳
                    self.best_avg_val_weighted_acc = avg_val_acc_w
                    self.best_acc_epoch = epoch
                elif avg_val_acc_w > self.second_best_avg_val_weighted_acc and avg_val_acc_w != self.best_avg_val_weighted_acc:
                    # 如果新值介于最佳和次佳之间 (且不等于最佳，以防重复记录)
                    self.second_best_avg_val_weighted_acc = avg_val_acc_w
                    self.second_best_acc_epoch = epoch
        else:
            self.logger.warning("纬度数据未加载，跳过验证轮次的加权指标计算。")
            # 如果没有纬度数据，avg_val_rmse_w 将保持 float('inf')

        # 保留原始的MAE和RMSE计算，如果需要的话
        y_pred_orig_metrics = y_pred.cpu() 
        y_true_orig_metrics = y_true.cpu()
        mask = (y_true_orig_metrics != 0.0) # 假设0是掩码值
        if torch.sum(mask).item() == 0:
            self.logger.info('  Validation MAE (original, masked): N/A (all values masked)')
            self._log_to_markdown('  Validation MAE (original, masked): N/A (all values masked)')
            self.logger.info('  Validation RMSE (original, masked): N/A (all values masked)')
            self._log_to_markdown('  Validation RMSE (original, masked): N/A (all values masked)')
        else:
            masked_true = y_true_orig_metrics[mask]
            masked_pred = y_pred_orig_metrics[mask]
            val_mae_orig = torch.abs(masked_pred - masked_true).mean()
            val_rmse_orig = torch.sqrt(((masked_pred - masked_true)**2).mean())
            self.logger.info('  Validation MAE (original, masked): {:.6f}'.format(val_mae_orig.item()))
            log_content_mae_orig = '  Validation MAE (original, masked): {:.6f}'.format(val_mae_orig.item())
            self._log_to_markdown(log_content_mae_orig)
            self.logger.info('  Validation RMSE (original, masked): {:.6f}'.format(val_rmse_orig.item()))
            log_content_rmse_orig = '  Validation RMSE (original, masked): {:.6f}'.format(val_rmse_orig.item())
            self._log_to_markdown(log_content_rmse_orig)
            
        return original_val_loss.item() # 或者返回包含 RMSE 的字典或元组，以便更方便地访问

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim].to(self.args.device)
            label = target[..., :self.args.output_dim].to(self.args.device)  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            
            # 启用R-Drop前向传播，获取两个不同的输出
            output1, output2 = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio, apply_r_drop=True)
            
            label = self.scaler.inverse_transform(label)[:,:,:,:1]

            if not self.args.real_value:
                # 对两个输出都进行逆变换
                output1_rescaled = self.scaler.inverse_transform(output1)[:,:,:,:1]
                output2_rescaled = self.scaler.inverse_transform(output2)[:,:,:,:1]
                # 计算监督损失
                loss_sup1 = self.loss(output1_rescaled.cuda(), label)
                loss_sup2 = self.loss(output2_rescaled.cuda(), label)
            else:
                # 直接使用模型输出计算监督损失
                loss_sup1 = self.loss(output1.cuda(), label)
                loss_sup2 = self.loss(output2.cuda(), label)
            
            # 平均监督损失
            loss_supervised = (loss_sup1 + loss_sup2) / 2
            
            # R-Drop正则化损失 (在模型输出空间计算)
            loss_reg = F.mse_loss(output1, output2)
            
            # 获取R-Drop系数beta并计算总损失
            beta = self.args.r_drop_beta
            loss = loss_supervised + beta * loss_reg

            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} (Sup: {:.6f}, Reg: {:.6f})'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), loss_supervised.item(), loss_reg.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
        log_content_train_loss = '**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio)
        self._log_to_markdown(log_content_train_loss)

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        # best_val_weighted_rmse = float('inf') # 旧的追踪变量，现在由 self.best_avg_val_weighted_rmse 代替
        # best_val_weighted_acc = float('-inf')  # 旧的追踪变量
        
        # 在每次调用 train 方法时（对应一个Optuna trial），重置最佳加权RMSE
        self.best_avg_val_weighted_rmse = float('inf')
        # 新增：每次训练前重置最优epoch
        self.best_avg_val_weighted_rmse_epoch = -1
        # 新增：用于记录最佳ACC的变量和对应的epoch
        self.best_avg_val_weighted_acc = float('-inf') # ACC越高越好
        self.best_acc_epoch = -1 # -1表示尚未获得
        # 新增：重置次佳指标
        self.second_best_avg_val_weighted_rmse = float('inf')
        self.second_best_avg_val_weighted_rmse_epoch = -1
        self.second_best_avg_val_weighted_acc = float('-inf')
        self.second_best_acc_epoch = -1
        
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        self.logger.info("Starting training process...") 
        if self.markdown_log_file_path:
            self._log_to_markdown("Trainer.train() method started. Test log entry.")
        else:
            self.logger.warning("Markdown logging is disabled due to initialization issues or path errors.")

        for epoch in range(1, self.args.epochs + 1): # 对于Optuna，这里只会循环一次 (epochs=1)
            train_epoch_loss = self.train_epoch(epoch)
            
            current_val_loss = float('inf')
            # current_avg_val_rmse_w = float('inf') # 不再需要局部变量
            # current_avg_val_acc_w = float('-inf') # 不再需要局部变量

            if self.val_loader != None:
                current_val_loss = self.val_epoch(epoch, self.val_loader)
                # self.best_avg_val_weighted_rmse 会在 val_epoch 内部更新
                # 新增：在每个epoch的验证后输出至今最优指标
                self.logger.info("---截至 Epoch {} 的最佳验证指标---".format(epoch))
                self._log_to_markdown("---截至 Epoch {} 的最佳验证指标---".format(epoch))
                if self.best_avg_val_weighted_rmse_epoch != -1:
                    self.logger.info("  - 最佳加权 RMSE: {:.6f} (Epoch {})"                     .format(self.best_avg_val_weighted_rmse, self.best_avg_val_weighted_rmse_epoch))
                    self._log_to_markdown("  - 最佳加权 RMSE: {:.6f} (Epoch {})" .format(self.best_avg_val_weighted_rmse, self.best_avg_val_weighted_rmse_epoch))
                else:
                    self.logger.info("  - 最佳加权 RMSE: N/A")
                    self._log_to_markdown("  - 最佳加权 RMSE: N/A")
                if self.best_acc_epoch != -1:
                    self.logger.info("  - 最佳加权 ACC:  {:.6f} (Epoch {})"                        .format(self.best_avg_val_weighted_acc, self.best_acc_epoch))
                    self._log_to_markdown("  - 最佳加权 ACC:  {:.6f} (Epoch {})" .format(self.best_avg_val_weighted_acc, self.best_acc_epoch))
                else:
                    self.logger.info("  - 最佳加权 ACC:  N/A")
                    self._log_to_markdown("  - 最佳加权 ACC:  N/A")
                # 新增：打印次佳指标
                if self.second_best_avg_val_weighted_rmse_epoch != -1:
                    self.logger.info("  - 次佳加权 RMSE: {:.6f} (Epoch {})" \
                                     .format(self.second_best_avg_val_weighted_rmse, self.second_best_avg_val_weighted_rmse_epoch))
                    self._log_to_markdown("  - 次佳加权 RMSE: {:.6f} (Epoch {})" .format(self.second_best_avg_val_weighted_rmse, self.second_best_avg_val_weighted_rmse_epoch))
                else:
                    self.logger.info("  - 次佳加权 RMSE: N/A")
                    self._log_to_markdown("  - 次佳加权 RMSE: N/A")
                if self.second_best_acc_epoch != -1:
                    self.logger.info("  - 次佳加权 ACC:  {:.6f} (Epoch {})" \
                                     .format(self.second_best_avg_val_weighted_acc, self.second_best_acc_epoch))
                    self._log_to_markdown("  - 次佳加权 ACC:  {:.6f} (Epoch {})" .format(self.second_best_avg_val_weighted_acc, self.second_best_acc_epoch))
                else:
                    self.logger.info("  - 次佳加权 ACC:  N/A")
                    self._log_to_markdown("  - 次佳加权 ACC:  N/A")
                self.logger.info("------------------------------------")
                self._log_to_markdown("------------------------------------")
            else: 
                current_val_loss = train_epoch_loss
                self.logger.info("警告: 未提供验证集，使用训练损失进行评估。此时 best_avg_val_weighted_rmse 将为 inf。")

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(current_val_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # 早停和最佳模型保存逻辑仍基于 current_val_loss
            # Optuna的剪枝将基于 self.best_avg_val_weighted_rmse (通过 trial.report)
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                not_improved_count = 0
                best_state = True
                self.logger.info(f'********************************* New best validation loss: {best_loss:.6f}')
                self._log_to_markdown(f'********************************* New best validation loss: {best_loss:.6f}')
            else:
                not_improved_count += 1
                best_state = False
            
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance (loss) didn't improve for {} epochs. Training stops.".format(self.args.early_stop_patience))
                    break
            
            if best_state == True:
                self.logger.info('Saving current best model (based on validation loss).')
                self._log_to_markdown('Saving current best model (based on validation loss).')
                best_model = copy.deepcopy(self.model.state_dict())
            
            # Optuna 剪枝相关的调用将在 Run.py 的 run_training_job 中进行
            # 因为 Run.py 可以访问 Optuna 的 trial 对象

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best validation loss: {:.6f}".format((training_time / 60), best_loss))
        if self.best_avg_val_weighted_rmse_epoch != -1:
            self.logger.info("Best validation average weighted RMSE for this trial: {:.6f} (achieved at epoch {})".format(self.best_avg_val_weighted_rmse, self.best_avg_val_weighted_rmse_epoch))
        else:
            self.logger.info("Best validation average weighted RMSE for this trial: {:.6f} (epoch not specifically tracked or N/A)".format(self.best_avg_val_weighted_rmse))
        
        # 新增：打印最终的次佳RMSE和最佳/次佳ACC
        if self.second_best_avg_val_weighted_rmse_epoch != -1:
            self.logger.info("Second best validation average weighted RMSE for this trial: {:.6f} (achieved at epoch {})" \
                             .format(self.second_best_avg_val_weighted_rmse, self.second_best_avg_val_weighted_rmse_epoch))
        
        if self.best_acc_epoch != -1: # 首先确保最佳ACC已记录
            self.logger.info("Best validation average weighted ACC for this trial: {:.6f} (achieved at epoch {})" \
                             .format(self.best_avg_val_weighted_acc, self.best_acc_epoch))
            if self.second_best_acc_epoch != -1:
                 self.logger.info("Second best validation average weighted ACC for this trial: {:.6f} (achieved at epoch {})" \
                                  .format(self.second_best_avg_val_weighted_acc, self.second_best_acc_epoch))


        if best_model is None and self.args.epochs > 0 : 
             self.logger.warning("No best model was saved during training (e.g. val loss never improved). Using the last state if available, or initial model.")
             best_model = copy.deepcopy(self.model.state_dict()) 
        elif best_model is None and self.args.epochs == 0:
            self.logger.error("No training epochs were run and no best model to load for testing.")
            # 对于Optuna单epoch场景，如果验证集不存在或出问题，best_model可能是None
            # 但我们主要关心 avg_weighted_rmse，模型本身是否保存对优化过程不是首要

        # 对于Optuna优化，通常不在每个trial结束时保存模型到文件，除非是最佳trial
        # 此处的保存逻辑可以保留，但优化脚本会管理最终最佳模型的保存
        if not self.args.debug and best_model is not None and self.args.epochs > 0 : # 添加 self.args.epochs > 0 条件
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current trial's best model to (log_dir): " + self.best_path)
            
            # 新增：保存到全局最佳模型路径
            global_save_folder_path = r"/root/autodl-tmp/AddPiData/Global-Best-Model"
            os.makedirs(global_save_folder_path, exist_ok=True)
            global_final_save_path = os.path.join(global_save_folder_path, 'best_model_{}.pth'.format(self.args.dataset))
            torch.save(best_model, global_final_save_path)
            self.logger.info(f"Additionally saving current trial's best model to global path: {global_final_save_path}")

        elif self.args.debug:
            self.logger.info("Debug mode: Best model not saved to file for this trial.")

        # --- 最终测试 --- 
        # 在Optuna的trial中，通常我们不运行最终测试，只关心验证集指标
        # 最终测试应该在所有trials完成后，使用找到的最佳参数进行一次
        # if best_model is not None:
        #     self.logger.info("Loading best model for final testing...")
        #     self.model.load_state_dict(best_model)
        #     self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, path_to_model_state=None) 
        # else:
        #     self.logger.warning("No best model state available for final testing for this trial.")

    def get_best_avg_val_weighted_rmse(self):
        """返回此 Trainer 实例在验证过程中记录的最佳（最小）平均加权RMSE。"""
        return self.best_avg_val_weighted_rmse

    # 新增：获取获得最优加权RMSE的epoch的方法
    def get_best_avg_val_weighted_rmse_epoch(self):
        """返回获得最佳加权RMSE的epoch编号。"""
        return self.best_avg_val_weighted_rmse_epoch

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def test(self, model, args, data_loader, scaler, logger, path_to_model_state=None):
        if path_to_model_state != None:
            logger.info(f"Loading model state from explicit path: {path_to_model_state}")
            try:
                check_point = torch.load(path_to_model_state, map_location=args.device)
                # 如果保存的是整个 Trainer 的 state, 或者只是 model state_dict
                if isinstance(check_point, dict) and 'state_dict' in check_point:
                    state_dict = check_point['state_dict']
                    # args_loaded = check_point['config'] # 如果需要加载旧的args
                else: # 假设直接是 state_dict
                    state_dict = check_point
                model.load_state_dict(state_dict)
            except Exception as e_load:
                logger.error(f"Error loading model from {path_to_model_state}: {e_load}")
                return
        # 如果 path_to_model_state is None, 则假定模型已在外部加载 (例如在 train 方法结束时)
        
        model.eval()
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim].to(args.device) # 确保数据在正确设备
                target = target.to(args.device) # 确保目标在正确设备
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0) # target 也应在设备上
                y_true_list.append(label.cpu()) # 收集到CPU以减少GPU内存占用
                y_pred_list.append(output.cpu())
        
        y_true_scaled = torch.cat(y_true_list, dim=0)
        y_pred_scaled = torch.cat(y_pred_list, dim=0)

        # 确保scaled张量在正确的设备上，以便scaler的逆转换能正确处理
        y_true_scaled = y_true_scaled.to(args.device)
        y_pred_scaled = y_pred_scaled.to(args.device)

        y_true = scaler.inverse_transform(y_true_scaled)[:,:,:,:args.output_dim]
        if args.real_value:
            y_pred = y_pred_scaled[:,:,:,:args.output_dim]
        else:
            y_pred = scaler.inverse_transform(y_pred_scaled)[:,:,:,:args.output_dim]

        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())

        # --- 计算原始指标 (MAE, RMSE, MAPE etc.) ---
        logger.info("Calculating original metrics (MAE, RMSE, MAPE) for test set...")
        for t_idx in range(y_true.shape[1]): # 遍历 horizon
            # All_Metrics 期望 (samples, nodes, features) or (samples, nodes)
            # y_pred/y_true 形状是 (B, H, N, D_out)
            # 取当前 horizon: (B, N, D_out) -> (B, N) if D_out=1
            pred_t = y_pred[:, t_idx, :, :].squeeze(-1) if args.output_dim == 1 else y_pred[:, t_idx, :, :] 
            true_t = y_true[:, t_idx, :, :].squeeze(-1) if args.output_dim == 1 else y_true[:, t_idx, :, :]

            mse, mae, rmse, mape, _, _ = All_Metrics(pred_t, true_t,
                                                args.mae_thresh, args.mape_thresh)
            logger.info("  Test Horizon {:02d}: MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t_idx + 1, mse, mae, rmse, mape*100 if mape is not None else -1))
        
        # 平均原始指标
        # All_Metrics 对整个y_pred, y_true (B,H,N,D_out) -> (B*H, N, D_out) or similar reshape
        # 或者分别计算然后平均。当前All_Metrics可能不直接支持4D输入后正确平均。
        # 简单起见，我们使用之前val_epoch中的masked MAE/RMSE逻辑对整个数据集（展平horizon）计算
        y_pred_flat_orig = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1]).squeeze(-1) if args.output_dim == 1 else y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
        y_true_flat_orig = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1]).squeeze(-1) if args.output_dim == 1 else y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
        
        mask_orig = (y_true_flat_orig != 0.0)
        if torch.sum(mask_orig).item() > 0:
            avg_mae_orig = torch.abs(y_pred_flat_orig[mask_orig] - y_true_flat_orig[mask_orig]).mean()
            avg_rmse_orig = torch.sqrt(((y_pred_flat_orig[mask_orig] - y_true_flat_orig[mask_orig])**2).mean())
            logger.info("  Test Avg Original: MAE (masked): {:.4f}, RMSE (masked): {:.4f}".format(avg_mae_orig.item(), avg_rmse_orig.item()))
        else:
            logger.info("  Test Avg Original: MAE/RMSE (masked): N/A (all values masked)")

        # --- 新增：计算加权 RMSE 和 ACC for Test set ---
        logger.info("Calculating weighted metrics (RMSE, ACC) for test set...")
        if self.latitudes is not None:
            latitude_weights = self.get_latitude_weights(self.latitudes.cpu()) # (N,)
            clim_for_acc = self.climatology_unnormalized.cpu()[:, :1].squeeze(-1) # (N,)
            if latitude_weights is None:
                 logger.warning("无法获取纬度权重，跳过测试集的加权指标计算。")
                 return

            test_rmse_weighted_list = []
            test_acc_weighted_list = []
            num_horizon_steps_test = y_pred.shape[1]

            for h_idx in range(num_horizon_steps_test):
                y_pred_h_test = y_pred[:, h_idx, :, 0].cpu() # (B, N)
                y_true_h_test = y_true[:, h_idx, :, 0].cpu() # (B, N)

                error_h_test = y_pred_h_test - y_true_h_test
                weighted_squared_error_h_test = (error_h_test**2) * latitude_weights.unsqueeze(0)
                current_rmse_w_test = torch.sqrt(weighted_squared_error_h_test.mean())
                test_rmse_weighted_list.append(current_rmse_w_test.item())
                logger.info('  Test Horizon {:02d}: Weighted RMSE: {:.6f}'.format(h_idx + 1, current_rmse_w_test.item()))

                pred_anom_h_test = y_pred_h_test - clim_for_acc.unsqueeze(0)
                true_anom_h_test = y_true_h_test - clim_for_acc.unsqueeze(0)
                pred_anom_prime_h_test = pred_anom_h_test - pred_anom_h_test.mean()
                true_anom_prime_h_test = true_anom_h_test - true_anom_h_test.mean()
                
                numerator_test = (pred_anom_prime_h_test * true_anom_prime_h_test * latitude_weights.unsqueeze(0)).sum()
                denominator_pred_sq_test = ((pred_anom_prime_h_test**2) * latitude_weights.unsqueeze(0)).sum()
                denominator_true_sq_test = ((true_anom_prime_h_test**2) * latitude_weights.unsqueeze(0)).sum()
                current_acc_w_test = numerator_test / (torch.sqrt(denominator_pred_sq_test * denominator_true_sq_test) + 1e-6)
                test_acc_weighted_list.append(current_acc_w_test.item())
                logger.info('  Test Horizon {:02d}: Weighted ACC:  {:.6f}'.format(h_idx + 1, current_acc_w_test.item()))
            
            if test_rmse_weighted_list:
                avg_test_rmse_w = np.mean(test_rmse_weighted_list)
                logger.info('  Test Avg: Weighted RMSE: {:.6f}'.format(avg_test_rmse_w))
            if test_acc_weighted_list:
                avg_test_acc_w = np.mean(test_acc_weighted_list)
                logger.info('  Test Avg: Weighted ACC:  {:.6f}'.format(avg_test_acc_w))
        else:
            logger.warning("纬度数据未加载，跳过测试集的加权指标计算。")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))