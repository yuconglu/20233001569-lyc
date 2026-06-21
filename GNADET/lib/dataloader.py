import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import pandas as pd
import os

def normalize_dataset(data, normalizer, column_wise=False, feature_wise = False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        elif feature_wise:
            mean = data.mean(axis=(0,1), keepdims=True)
            std = data.std(axis=(0,1), keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def split_data_by_numbers(data, val_ratio, test_ratio):
    test_data = data[-int(test_ratio):]
    val_data = data[-int((test_ratio+val_ratio)):-int(test_ratio)]
    train_data = data[:-int((test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D (T, N, F)
    print(f"原始数据加载完成，形状: {data.shape}")

    # --- 确定基础数据路径 (base_data_path) --- 
    base_data_path = None

    # 0. 最优先使用 args.data_dir (如果通过 Run.py 从 GIS.conf 传入且有效)
    if hasattr(args, 'data_dir') and args.data_dir and os.path.isdir(args.data_dir):
        base_data_path = args.data_dir
        print(f"使用来自 args.data_dir (源自GIS.conf) 的路径: {base_data_path}")

    # 1. 其次，尝试使用 args.processed_data_dir (如果提供且有效，并且尚未通过 args.data_dir 设置)
    if base_data_path is None and hasattr(args, 'processed_data_dir') and args.processed_data_dir and os.path.isdir(args.processed_data_dir):
        base_data_path = args.processed_data_dir
        print(f"使用提供的 processed_data_dir: {base_data_path}")
    
    # 2. 如果没有，尝试从 args.dataset 推断 (如果它是文件路径，并且尚未设置 base_data_path)
    if base_data_path is None and os.path.isfile(args.dataset):
        base_data_path = os.path.dirname(args.dataset)
        print(f"从 args.dataset 文件路径推断的目录: {base_data_path}")

    # 3. 如果还是没有，并且 args.dataset 是一个存在的目录 (并且尚未设置 base_data_path)
    if base_data_path is None and os.path.isdir(args.dataset):
        base_data_path = args.dataset # args.dataset 本身就是目录
        print(f"使用 args.dataset 作为目录: {base_data_path}")

    # 4. 如果以上都不行，回退到硬编码的已知正确路径
    if base_data_path is None or not os.path.isdir(base_data_path):
        fallback_path = r"C:\Users\30922\Desktop\AddPiData\BestGIS\NADE-main\data"
        print(f"警告: 无法从参数确定有效的数据目录，将回退到默认路径: {fallback_path}")
        base_data_path = fallback_path
        if not os.path.isdir(base_data_path):
            print(f"错误: 回退路径 {base_data_path} 也不存在或不是一个目录。请检查路径配置。")
            exit()

    time_coords_filename = 'time_coords_global_6H_2006_2018.npy'
    time_coords_path = os.path.join(base_data_path, time_coords_filename)
    
    try:
        print(f"加载时间坐标文件: {time_coords_path}")
        time_coords = np.load(time_coords_path)
        if not np.issubdtype(time_coords.dtype, np.datetime64):
            time_coords = time_coords.astype('datetime64[ns]')
        time_index = pd.to_datetime(time_coords)
        print(f"时间坐标加载成功，范围: {time_index.min()} to {time_index.max()}")
        
        # 确保加载的数据data的时间维度长度与时间坐标一致
        if data.shape[0] != len(time_index):
            print(f"错误: 加载的数据时间维度 ({data.shape[0]}) 与时间坐标长度 ({len(time_index)}) 不匹配!")
            print("请确保 X_GIS.npy (或其他由args.dataset指定的文件) 和 time_coords_global_6H_2006_2018.npy 是对应的。")
            exit()

    except FileNotFoundError:
        print(f"错误: 时间坐标文件 {time_coords_path} 未找到。无法按年份划分数据集。")
        exit()
    except Exception as e_time_load:
        print(f"加载或处理时间坐标时出错: {e_time_load}")
        exit()

    #normalize st data (保持在划分前，按您的要求尽量少改动)
    data, scaler = normalize_dataset(data, normalizer, args.column_wise, args.feature_wise)

    # 按年份划分数据集 (在标准化之后，使用原始的未标准化数据的掩码)
    print("按年份划分数据集...")
    train_mask = (time_index >= pd.Timestamp('2006-01-01')) & (time_index < pd.Timestamp('2016-01-01'))
    val_mask = (time_index >= pd.Timestamp('2016-01-01')) & (time_index < pd.Timestamp('2017-01-01'))
    test_mask = (time_index >= pd.Timestamp('2017-01-01')) & (time_index < pd.Timestamp('2019-01-01'))

    data_train = data[train_mask]
    data_val = data[val_mask]
    data_test = data[test_mask]

    print(f"划分后形状: Train={data_train.shape}, Val={data_val.shape}, Test={data_test.shape}")
    if data_train.shape[0] == 0 or data_test.shape[0] == 0:
        print("警告：按年份划分后，训练集或测试集为空！请检查时间范围、数据和时间坐标的对应关系。")
        # 允许验证集为空，根据原始逻辑
        if data_train.shape[0] == 0 and data_test.shape[0] == 0:
            exit("错误：训练集和测试集均为空，无法继续。")

    # --- 计算气候学数据 ---
    # 首先，需要获取未标准化的训练数据部分的真实目标变量
    # 我们假设原始的 `data` 在标准化前是未标准化的。
    # 我们需要重新加载原始数据，或者如果 `scaler.inverse_transform` 可靠，用它来逆转换标准化后的训练数据部分。
    # 为了简单和准确，我们重新基于 mask 从原始未缩放数据计算 climatology
    
    print("加载原始数据以计算气候学数据...")
    raw_data_for_clim = load_st_dataset(args.dataset) # 这会重新加载未缩放的原始数据 T, N, F
    
    # 使用训练数据的掩码从原始数据中提取训练部分
    raw_train_data_for_clim = raw_data_for_clim[train_mask]

    climatology_unnormalized = None
    if raw_train_data_for_clim.shape[0] > 0:
        # 假设目标变量是第一个特征 (或者由 args.output_dim 定义的切片)
        # 并且我们只需要第一个输出维度进行气候学计算 (如果 output_dim > 1)
        # output_dim 通常为1，所以我们取 :1
        climatology_unnormalized = raw_train_data_for_clim[:, :, :args.output_dim].mean(axis=0) # 平均时间维度 (T_train, N, D_out) -> (N, D_out)
        print(f"计算得到的气候学数据形状 (未标准化): {climatology_unnormalized.shape}")
    else:
        print("警告: 训练数据为空，无法计算气候学数据。将使用零数组代替。")
        # 如果训练数据为空，创建一个形状正确但值为零的 climatology
        # 这在实际应用中应该避免，因为气候学数据对ACC计算至关重要
        climatology_unnormalized = np.zeros((args.num_nodes, args.output_dim))

    # --- 在这里添加时间特征处理 ---
    if args.time_dependence:
        print("正在添加时间依赖特征...")
        
        # 提取与数据子集对应的时间索引
        time_index_train = time_index[train_mask]
        time_index_val = time_index[val_mask]
        time_index_test = time_index[test_mask]

        def get_time_feature(times_idx, num_nodes, data_subset_shape_0):
            """
            根据时间索引计算时间特征 (一天中的第几个6小时间隔)。
            确保生成的时间特征的时间步数与对应的数据子集匹配。
            """
            if len(times_idx) != data_subset_shape_0:
                print(f"警告: 传入 get_time_feature 的时间索引长度 ({len(times_idx)}) 与数据子集的时间步数 ({data_subset_shape_0}) 不匹配。将尝试截取或填充时间索引以匹配数据。")
                # 简单的处理：如果时间索引更长，则截断；如果更短，这里可能需要更复杂的逻辑或报错
                # 为了避免在concatenate时出错，这里我们确保长度一致，但实际应用中应仔细检查数据源
                if len(times_idx) > data_subset_shape_0:
                    times_idx = times_idx[:data_subset_shape_0]
                else:
                    # 如果时间索引比数据短，这是一个更严重的问题，可能表示数据或掩码逻辑错误
                    # 这里我们先打印错误并返回None，让后续的concatenate失败，以便用户注意到
                    print(f"错误: 时间索引长度 ({len(times_idx)}) 小于数据子集时间步数 ({data_subset_shape_0}). 无法安全地生成时间特征。")
                    return None


            feature = (times_idx.hour // 6).to_numpy() # (时间步数,)
            # 归一化到0-1 (除以最大可能值3，因为 0, 6, 12, 18小时对应索引 0, 1, 2, 3)
            feature = feature / 3.0 
            # 扩展维度以匹配 (时间步数, 节点数, 1)
            feature_expanded = np.repeat(feature[:, np.newaxis], num_nodes, axis=1) # (时间步数, 节点数)
            feature_expanded = feature_expanded[..., np.newaxis] # (时间步数, 节点数, 1)
            return feature_expanded

        if data_train.shape[0] > 0 :
            time_feat_train = get_time_feature(time_index_train, args.num_nodes, data_train.shape[0])
            if time_feat_train is not None and data_train.shape[0] == time_feat_train.shape[0]:
                data_train = np.concatenate((data_train, time_feat_train), axis=-1)
            elif time_feat_train is None:
                 print("错误：未能为训练集生成时间特征。")   
            else:
                print(f"警告: 训练数据长度 ({data_train.shape[0]}) 与生成的时间特征长度 ({time_feat_train.shape[0]}) 不匹配！拼接操作可能失败或导致错误。")
        
        if data_val.shape[0] > 0: # 仅当验证集非空时处理
            time_feat_val = get_time_feature(time_index_val, args.num_nodes, data_val.shape[0])
            if time_feat_val is not None and data_val.shape[0] == time_feat_val.shape[0]:
                data_val = np.concatenate((data_val, time_feat_val), axis=-1)
            elif time_feat_val is None:
                print("错误：未能为验证集生成时间特征。")
            else:
                print(f"警告: 验证数据长度 ({data_val.shape[0]}) 与生成的时间特征长度 ({time_feat_val.shape[0]}) 不匹配！拼接操作可能失败或导致错误。")

        if data_test.shape[0] > 0:
            time_feat_test = get_time_feature(time_index_test, args.num_nodes, data_test.shape[0])
            if time_feat_test is not None and data_test.shape[0] == time_feat_test.shape[0]:
                data_test = np.concatenate((data_test, time_feat_test), axis=-1)
            elif time_feat_test is None:
                print("错误：未能为测试集生成时间特征。")
            else:
                print(f"警告: 测试数据长度 ({data_test.shape[0]}) 与生成的时间特征长度 ({time_feat_test.shape[0]}) 不匹配！拼接操作可能失败或导致错误。")
        
        print(f"添加时间特征后形状: Train={data_train.shape}, Val={data_val.shape}, Test={data_test.shape}")
    # --- 时间特征处理结束 ---

    #spilit dataset by days or by ratio (这部分逻辑被新的按年份划分取代)
    # if args.time_dependence : 
    # if args.test_ratio > 1:
    #     data_train, data_val, data_test = split_data_by_numbers(data, args.val_ratio, args.test_ratio)
    # else:
    #     data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single, args.dataset)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single, args.dataset)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single, args.dataset)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, climatology_unnormalized

    # train_dataloader = data_loader(x_tra, y_tra, x_tra.shape[0], shuffle=True, drop_last=True)
    # if len(x_val) == 0:
    #     val_dataloader = None
    # else:
    #     val_dataloader = data_loader(x_val, y_val, x_val.shape[0], shuffle=False, drop_last=True)
    # test_dataloader = data_loader(x_test, y_test, x_test.shape[0], shuffle=False, drop_last=False)
    # return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    # parser.add_argument('--val_ratio', default=0.1, type=float) # 已被年份划分取代
    # parser.add_argument('--test_ratio', default=0.2, type=float) # 已被年份划分取代
    # 新增一个参数来指定包含 time_coords_global_6H_2006_2018.npy 的目录
    parser.add_argument('--processed_data_dir', default=None, type=str, 
                        help='Path to the directory containing X_GIS.npy (or equivalent) and time_coords_global_6H_2006_2018.npy')

    parser.add_argument('--column_wise', action='store_true', help='normalize column-wise')
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--time_dependence', action='store_true', help='add time dependence')
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)