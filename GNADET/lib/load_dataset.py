import os
import numpy as np
import torch
import csv

def load_st_dataset(dataset):
    #output B, N, D
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.dirname(current_dir)
    # 构建数据文件的绝对路径
    data_dir = os.path.join(root_dir, 'data')
    
    if dataset == 'LA':
        data_path = os.path.join(data_dir, 'X_LA.npy')
        data = np.load(data_path)    
    elif dataset == 'SD':
        data_path = os.path.join(data_dir, 'X_SD.npy')
        data = np.load(data_path)
    elif dataset == 'NOAA':
        data_path = os.path.join(data_dir, 'X_NOAA.npy')
        data = np.load(data_path)
    elif dataset == 'GIS':
        # 优先使用 Spark 预处理后的数据，没有则用原始数据
        spark_path = os.path.join(data_dir, 'X_GIS_spark.npy')
        data_path = spark_path if os.path.exists(spark_path) else os.path.join(data_dir, 'X_GIS.npy')
        data = np.load(data_path)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def get_adjacency_matrix(args):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    edge_index: torch.Tensor, 边索引张量，用于GCN
    '''

    if args.dataset in ['LA', 'SD', 'NOAA', 'GIS']:
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录
        root_dir = os.path.dirname(current_dir)
        # 构建数据文件的绝对路径
        data_dir = os.path.join(root_dir, 'data')

        # --- 根据数据集名称选择正确的 edge_index 文件 ---
        if args.dataset == 'GIS':
            edge_path = os.path.join(data_dir, 'edge_index_GIS.npy')
            print(f"加载 GIS 邻接矩阵: {edge_path}")
        else:
            # 保留原始逻辑加载 LA, SD, NOAA 的邻接矩阵
            edge_path = os.path.join(data_dir, 'edge_index_{}.npy'.format(args.dataset))
            print(f"加载 {args.dataset} 邻接矩阵: {edge_path}")
        # --- 结束选择 ---

        edge_idx = np.load(edge_path)
        
        # 直接返回边索引张量，用于GCN
        edge_index = torch.tensor(edge_idx, dtype=torch.int64).to(args.device)
        return edge_index

    else:
        raise ValueError