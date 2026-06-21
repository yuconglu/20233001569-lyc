import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.NADE import Net as Network

from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
import optuna


#*************************************************************************#
Mode = 'train'
DEBUG = False
DATASET = 'GIS'    
DEVICE = 'cuda'
MODEL = 'NADE'

#get configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, '{}.conf'.format(DATASET))
print('读取配置文件: %s' % (config_file))
config = configparser.ConfigParser()
# config.read(config_file) # 原始读取方式
# 指定使用 utf-8 编码读取配置文件，以支持中文注释
config.read(config_file, encoding='utf-8')

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def run_training_job(trial=None):
    """执行一次完整的训练和评估流程，并返回验证集上的平均加权RMSE。
    Args:
        trial (optuna.trial.Trial, optional): Optuna的trial对象，用于剪枝。
    Returns:
        float: 验证集上的平均加权RMSE。
    """
    #*************************************************************************#
    Mode = 'train' # 始终在训练模式下运行以进行优化
    DEBUG = False # 可以根据需要调整，或者也作为参数
    DATASET = 'GIS'    
    DEVICE = 'cuda' # 优先使用cuda
    MODEL = 'NADE'

    #get configuration
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用相对路径，假设脚本从项目根目录的上一级或者特定工作目录执行
    # 如果 optimize.py 在 NADE-main 目录下，那么 GIS.conf 的路径应该是 'model/GIS.conf'
    # 当前 Run.py 在 model 目录下，GIS.conf 也在 model 目录下
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}.conf'.format(DATASET))
    
    print(f'读取配置文件: {config_file_path}')
    config = configparser.ConfigParser()
    config.read(config_file_path, encoding='utf-8')

    #parser
    args = argparse.ArgumentParser(description='arguments')
    # 从配置文件加载的参数，这里保持不变，因为 embed_dim 和 hidden_dim 将直接在 args 对象上被覆盖
    # (或者，如果坚持方案A的纯粹文件修改，则不需要在这里覆盖，而是在Optuna的objective函数中修改文件)
    # 为了将 trial 对象传递并使用其参数，我们这里假设 embed_dim 和 hidden_dim 可以从 trial 获取并设置到 args
    # 但计划是直接修改文件，所以这里不需要从trial获取，而是从已修改的conf文件读取。

    args.add_argument('--dataset', default=DATASET, type=str, help="数据集名称 (例如 'GIS', 'NOAA', 'LA', 'SD')")
    args.add_argument('--mode', default=Mode, type=str)
    args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
    args.add_argument('--debug', default=DEBUG, type=eval)
    args.add_argument('--model', default=MODEL, type=str)
    args.add_argument('--cuda', default=True, type=bool)
    #data
    args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('--lag', default=config['data']['lag'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--tod', default=config['data']['tod'], type=eval)
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('--feature_wise', default=config['data']['feature_wise'], type=eval)
    # 新增 data_dir 参数，从配置文件读取
    args.add_argument('--data_dir', default=config['data'].get('data_dir', None), type=str, help="包含已处理数据和辅助文件（如time_coords和lat_coords）的目录")
    #model
    # embed_dim 和 hidden_dim 将从 (可能被Optuna修改过的) GIS.conf 文件中读取
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int) 
    args.add_argument('--hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('--alpha', default=config['model']['alpha'], type=float)
    args.add_argument('--time_dependence', default=config['model']['time_dependence'], type=eval)
    args.add_argument('--time_divided', default=config['model']['time_divided'], type=eval)
    args.add_argument('--model_type', default=config['model']['model_type'], type=str)

    #train
    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int) # 应为1
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--teacher_forcing', default=False, type=bool)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    #test
    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    #log
    args.add_argument('--log_dir', default='./', type=str)
    args.add_argument('--log_step', default=config['log']['log_step'], type=int)
    args.add_argument('--plot', default=config['log']['plot'], type=eval)
    # R-Drop 相关参数
    args.add_argument('--dropout_rate', type=float, default=config['train'].getfloat('dropout_rate', 0.1), help='dropout rate for R-Drop') # 从配置文件读取
    args.add_argument('--r_drop_beta', type=float, default=config['train'].getfloat('r_drop_beta', 1.0), help='R-Drop 正则化系数 beta') # 修改 alpha 为 beta，并从配置文件读取 r_drop_beta
    # Transformer 相关参数
    args.add_argument('--transformer_heads', type=int, default=config['model'].getint('transformer_heads', 4), help='number of heads in transformer') # 从配置文件读取
    args.add_argument('--transformer_dropout', type=float, default=config['model'].getfloat('transformer_dropout', 0.1), help='dropout in transformer') # 从配置文件读取
    
    # 解析参数，注意：如果 optimize.py 多次调用此函数，argparse状态可能需要小心处理
    # 一个更稳健的方法是直接从config对象创建args的命名空间或字典，而不是重复调用parse_args()
    # 或者，每次都创建一个新的ArgumentParser实例
    # args = args.parse_args() # 原始方式
    # 为了避免多次调用parse_args()可能引发的问题（特别是在同一个Python进程中多次调用时），
    # 我们可以解析一次，或者更好的是，直接从config构建一个简单的命名空间对象。
    # 这里我们暂时保留原始的 argparse 结构，但需注意 optimize.py 中对 Run.py 的调用方式。
    # 如果 optimize.py 是通过 subprocess 调用 Run.py，则每次都是新进程，没问题。
    # 如果是函数调用，则需要确保 argparse 不会因为重复定义参数而出错。
    # 为简化，假定 optimize.py 会以某种方式确保这里的 parse_args() 每次都正确运行。
    # 或者，更简单地，因为所有值都来自config，我们可以手动构建一个args对象。 
    # 让我们尝试手动构建args，以避免parse_args的潜在问题
    parsed_args = args.parse_known_args()[0] # 解析已定义的参数，忽略未知的

    init_seed(parsed_args.seed)

    if torch.cuda.is_available() and parsed_args.cuda:
        parsed_args.device = 'cuda'
    else:
        parsed_args.device = 'cpu'

    if parsed_args.time_dependence:
        parsed_args.input_dim = parsed_args.input_dim + 1
        print(f"启用时间依赖特征，模型输入维度调整为: {parsed_args.input_dim}")

    #load dataset
    train_loader, val_loader, test_loader, scaler, climatology_unnormalized = get_dataloader(parsed_args,
                                                                   normalizer=parsed_args.normalizer,
                                                                   tod=parsed_args.tod, dow=False,
                                                                   weather=False, single=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) # model目录的上一级，即NADE-main
    # data_dir = os.path.join(root_dir, 'data') # 这个data_dir似乎没有被直接使用

    from lib.load_dataset import get_adjacency_matrix
    edge_index = get_adjacency_matrix(parsed_args)

    #init model
    # 注意：这里的 embed_dim 和 hidden_dim 将会是 Optuna 修改后从 conf 文件读取的值
    model = Network(parsed_args, edge_index)
    model = model.to(parsed_args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    if parsed_args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif parsed_args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(parsed_args.device)
    elif parsed_args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(parsed_args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=parsed_args.lr_init, eps=1.0e-8,
                                 weight_decay=0.0005, amsgrad=False)
    lr_scheduler = None
    if parsed_args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(parsed_args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=parsed_args.lr_decay_rate)

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # 确保日志目录对于每个trial是唯一的，或者由Optuna管理
    # 为了简化，这里可以使用固定的log_dir，或者基于trial id创建子目录
    # 如果Optuna多次调用此函数，日志会写到同一个地方，或者需要更精细的控制
    # log_dir_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', parsed_args.dataset)
    # trial_id_str = f"trial_{trial.number}" if trial else "default"
    # log_dir = os.path.join(log_dir_base, trial_id_str, current_time) # 为每个trial创建唯一日志路径
    # os.makedirs(log_dir, exist_ok=True)
    # parsed_args.log_dir = log_dir
    # 简化：使用原始的日志路径逻辑，但注意并发写入问题（如果n_jobs > 1）
    # 或者，让 Optuna 的 logger 处理日志，并减少这里的日志量级
    log_dir_path_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', parsed_args.dataset)
    if trial:
        # 为每个Optuna trial创建一个独特的日志目录，以避免日志覆盖
        trial_log_dir = os.path.join(log_dir_path_base, f"trial_{trial.number}_{current_time}")
    else:
        trial_log_dir = os.path.join(log_dir_path_base, current_time)
    os.makedirs(trial_log_dir, exist_ok=True)
    parsed_args.log_dir = trial_log_dir
    

    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      parsed_args, climatology_unnormalized, lr_scheduler=lr_scheduler)

    avg_val_weighted_rmse_for_trial = float('inf')

    if parsed_args.mode == 'train':
        trainer.train() # train方法内部会进行一轮训练和验证
        avg_val_weighted_rmse_for_trial = trainer.get_best_avg_val_weighted_rmse()
        # 新增：获取获得最优加权RMSE的epoch
        best_rmse_epoch = trainer.get_best_avg_val_weighted_rmse_epoch()
        # 输出最优指标和获得该指标的epoch
        print(f"Best validation average weighted RMSE for this trial: {avg_val_weighted_rmse_for_trial:.6f}")
        print(f"Best RMSE was achieved at epoch: {best_rmse_epoch}")  # 中文注释：输出获得最优加权RMSE的轮次
        # 详细中文注释：上述print语句会在训练结束后，输出本次trial获得的最优加权RMSE及其出现的epoch（轮次）
        
        # Optuna 剪枝逻辑重新添加
        if trial: # 只有在 Optuna 调用时 trial 对象才存在
            current_epoch_for_report = 1 # 因为我们只训练一个epoch (args.epochs 应该为1)
            trial.report(avg_val_weighted_rmse_for_trial, step=current_epoch_for_report) 
            if trial.should_prune():
                # print(f"Trial {trial.number} pruned at epoch {current_epoch_for_report} with RMSE: {avg_val_weighted_rmse_for_trial}") # Optuna会自动记录
                raise optuna.TrialPruned()
    
    elif parsed_args.mode == 'test':
        # Optuna 优化时不执行测试模式
        print("警告: 在Optuna优化期间不应执行测试模式。")
        # 可以加载预训练模型并测试，但这不属于优化循环的一部分
        # pretrained_dir = os.path.join(root_dir, 'pre-trained')
        # model_path = os.path.join(pretrained_dir, '{}.pth'.format(parsed_args.dataset))
        # if os.path.exists(model_path):
        #     print("加载预训练模型: {}".format(model_path))
        #     try:
        #         model.load_state_dict(torch.load(model_path, map_location=parsed_args.device))
        #     except RuntimeError as e:
        #         print(f"加载模型状态字典时出错: {e}")
        #         model.load_state_dict(torch.load(model_path, map_location='cpu'))
        #         model.to(parsed_args.device)
        #     trainer.test(model, parsed_args, test_loader, scaler, trainer.logger, path=None)
        # else:
        #     print(f"错误：找不到预训练模型文件 {model_path}，无法执行测试模式。")
    else:
        raise ValueError("未知模式: {}".format(parsed_args.mode))

    return avg_val_weighted_rmse_for_trial

if __name__ == '__main__':
    # 原始的执行逻辑，当直接运行Run.py时触发
    # 对于Optuna，这个部分不会被 optimize.py 调用
    # 但可以保留用于单独测试 Run.py 的功能
    print("Run.py executed as main script.")
    # run_training_job() #可以调用它进行一次默认的训练，但不带trial对象
    # 或者更复杂的逻辑来处理命令行参数，然后调用run_training_job
    # 为了简单，我们暂时只打印信息，让 optimize.py 作为主要入口

    # 为了使原始的 argparse 生效，我们需要这样设置：
    # get configuration
    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    DATASET_main = 'GIS' # 或者从实际参数获取
    config_file_main = os.path.join(script_dir_main, '{}.conf'.format(DATASET_main))
    print('读取配置文件 (main): %s' % (config_file_main))
    config_main = configparser.ConfigParser()
    config_main.read(config_file_main, encoding='utf-8')

    args_main_parser = argparse.ArgumentParser(description='arguments')
    args_main_parser.add_argument('--dataset', default=DATASET_main, type=str)
    args_main_parser.add_argument('--mode', default='train', type=str)
    args_main_parser.add_argument('--device', default='cuda', type=str)
    args_main_parser.add_argument('--debug', default=True, type=eval)
    args_main_parser.add_argument('--model', default='NADE', type=str)
    args_main_parser.add_argument('--cuda', default=True, type=bool)
    args_main_parser.add_argument('--val_ratio', default=config_main['data']['val_ratio'], type=float)
    args_main_parser.add_argument('--test_ratio', default=config_main['data']['test_ratio'], type=float)
    args_main_parser.add_argument('--lag', default=config_main['data']['lag'], type=int)
    args_main_parser.add_argument('--horizon', default=config_main['data']['horizon'], type=int)
    args_main_parser.add_argument('--num_nodes', default=config_main['data']['num_nodes'], type=int)
    args_main_parser.add_argument('--tod', default=config_main['data']['tod'], type=eval)
    args_main_parser.add_argument('--normalizer', default=config_main['data']['normalizer'], type=str)
    args_main_parser.add_argument('--column_wise', default=config_main['data']['column_wise'], type=eval)
    args_main_parser.add_argument('--feature_wise', default=config_main['data']['feature_wise'], type=eval)
    args_main_parser.add_argument('--input_dim', default=config_main['model']['input_dim'], type=int)
    args_main_parser.add_argument('--output_dim', default=config_main['model']['output_dim'], type=int)
    args_main_parser.add_argument('--embed_dim', default=config_main['model']['embed_dim'], type=int)
    args_main_parser.add_argument('--hidden_dim', default=config_main['model']['hidden_dim'], type=int)
    args_main_parser.add_argument('--alpha', default=config_main['model']['alpha'], type=float)
    args_main_parser.add_argument('--time_dependence', default=config_main['model']['time_dependence'], type=eval)
    args_main_parser.add_argument('--time_divided', default=config_main['model']['time_divided'], type=eval)
    args_main_parser.add_argument('--model_type', default=config_main['model']['model_type'], type=str)
    args_main_parser.add_argument('--loss_func', default=config_main['train']['loss_func'], type=str)
    args_main_parser.add_argument('--seed', default=config_main['train']['seed'], type=int)
    args_main_parser.add_argument('--batch_size', default=config_main['train']['batch_size'], type=int)
    args_main_parser.add_argument('--epochs', default=config_main['train']['epochs'], type=int)
    args_main_parser.add_argument('--lr_init', default=config_main['train']['lr_init'], type=float)
    args_main_parser.add_argument('--lr_decay', default=config_main['train']['lr_decay'], type=eval)
    args_main_parser.add_argument('--lr_decay_rate', default=config_main['train']['lr_decay_rate'], type=float)
    args_main_parser.add_argument('--lr_decay_step', default=config_main['train']['lr_decay_step'], type=str)
    args_main_parser.add_argument('--early_stop', default=config_main['train']['early_stop'], type=eval)
    args_main_parser.add_argument('--early_stop_patience', default=config_main['train']['early_stop_patience'], type=int)
    args_main_parser.add_argument('--grad_norm', default=config_main['train']['grad_norm'], type=eval)
    args_main_parser.add_argument('--max_grad_norm', default=config_main['train']['max_grad_norm'], type=int)
    args_main_parser.add_argument('--teacher_forcing', default=False, type=bool)
    args_main_parser.add_argument('--real_value', default=config_main['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    args_main_parser.add_argument('--mae_thresh', default=config_main['test']['mae_thresh'], type=eval)
    args_main_parser.add_argument('--mape_thresh', default=config_main['test']['mape_thresh'], type=float)
    args_main_parser.add_argument('--log_dir', default='./', type=str)
    args_main_parser.add_argument('--log_step', default=config_main['log']['log_step'], type=int)
    args_main_parser.add_argument('--plot', default=config_main['log']['plot'], type=eval)
    # R-Drop 相关参数
    args_main_parser.add_argument('--dropout_rate', type=float, default=config_main['train'].getfloat('dropout_rate', 0.1), help='dropout rate for R-Drop') # 从配置文件读取
    args_main_parser.add_argument('--r_drop_beta', type=float, default=config_main['train'].getfloat('r_drop_beta', 1.0), help='R-Drop 正则化系数 beta') # 修改 alpha 为 beta，并从配置文件读取 r_drop_beta
    # Transformer 相关参数
    args_main_parser.add_argument('--transformer_heads', type=int, default=config_main['model'].getint('transformer_heads', 4), help='number of heads in transformer') # 从配置文件读取
    args_main_parser.add_argument('--transformer_dropout', type=float, default=config_main['model'].getfloat('transformer_dropout', 0.1), help='dropout in transformer') # 从配置文件读取
    
    # global args # 声明 args 为全局变量，以便 run_training_job 可以访问
    args = args_main_parser.parse_args() # 解析命令行参数，填充到 args 中
    
    # 将args传递给run_training_job，或让run_training_job内部重新定义和解析
    # 当前的run_training_job会自己定义和解析，所以这里不需要直接传递args
    # 如果要使用这里解析的args，需要修改run_training_job的参数和内部逻辑
    # run_training_job() # 调用时不带 trial，将使用默认参数运行

    # 最简单的处理方式是，如果直接运行Run.py，则执行一次默认的训练
    print("Running a default training job as __main__...")
    default_rmse = run_training_job(trial=None) # trial is None for a default run
    print(f"Default training job finished. Validation Avg Weighted RMSE: {default_rmse}")
