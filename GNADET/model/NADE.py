import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric.nn as pyg_nn

from torchdiffeq import odeint_adjoint as odeint

class ODEfunc(nn.Module):
    
    def __init__(self, edge_index, model_type, num_nodes, latent_dim, nhidden, alpha, embed_dim, transformer_heads, transformer_dropout):
        super(ODEfunc, self).__init__()

        self.model_type = model_type
        self.alpha = alpha
        self.embed_dim = embed_dim
        self.edge_index = edge_index
        
        # 为了保持与原始代码中物理项计算的兼容性
        # 创建一个全零的邻接矩阵以便后续填充
        self.adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        
        self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.A2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        if self.model_type == 'k':
            self.coeff = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))

        # 定义 TransformerConv 层
        self.transformer_conv = pyg_nn.TransformerConv(
            in_channels=latent_dim,
            out_channels=latent_dim,
            heads=transformer_heads,
            dropout=transformer_dropout,
            concat=False # 保持输出维度 = latent_dim
        )

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        if self.model_type == 'diff':
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A1.T)))
        elif self.model_type == 'adv':
            A_out = F.relu(torch.tanh(self.alpha * (torch.mm(self.A1, self.A2.T) - torch.mm(self.A2, self.A1.T))))
        else: 
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A2.T)))

        if self.model_type == 'pre':
            # 当需要预设邻接矩阵时，从edge_index创建一个稠密矩阵
            indices = self.edge_index
            values = torch.ones(indices.size(1), device=indices.device)
            size = (self.A1.size(0), self.A1.size(0))
            A_out = torch.sparse.FloatTensor(indices, values, size).to_dense()
        elif self.model_type == 'k':
            # 从edge_index创建一个稠密矩阵
            indices = self.edge_index
            values = torch.ones(indices.size(1), device=indices.device)
            size = (self.A1.size(0), self.A1.size(0))
            self.adj = torch.sparse.FloatTensor(indices, values, size).to_dense()
            A_out = self.coeff * self.adj
        else:
            # 从edge_index创建一个稠密矩阵用于物理项
            indices = self.edge_index
            values = torch.ones(indices.size(1), device=indices.device)
            size = (self.A1.size(0), self.A1.size(0))
            self.adj = torch.sparse.FloatTensor(indices, values, size).to_dense()
            A_out = A_out * self.adj

        D_out = torch.diag(A_out.sum(1))
        L = D_out - A_out.T
      
        # 用TransformerConv替换原有的GCN
        batch_size = x.size(0)
        # 展平批次维度
        transformer_input = x.view(-1, x.size(-1)) 
        
        # 应用TransformerConv层
        uncertainty_flat = self.transformer_conv(transformer_input, self.edge_index)
        # 应用激活函数
        uncertainty_flat = F.gelu(uncertainty_flat)
        
        # 恢复原始形状
        uncertainty = uncertainty_flat.view(batch_size, -1, uncertainty_flat.size(-1))

        if self.model_type == 'onlyf':
            return uncertainty
        elif self.model_type == 'withoutf':
            physical_term = -(torch.matmul(L, x))
            return physical_term
        else:
            physical_term = -(torch.matmul(L, x))
            return uncertainty + physical_term


class ODEBlock(nn.Module):
    def __init__(self,
                edge_index,
                model_type,
                num_nodes, 
                in_features,
                horizon,
                alpha,
                embed_dim,
                transformer_heads,
                transformer_dropout,
                method='dopri5',
                adjoint=True,
                atol=1e-3,
                rtol=1e-3):
        super(ODEBlock, self).__init__()


        self.odefunc = ODEfunc(edge_index, model_type, num_nodes, in_features, in_features*4, alpha, embed_dim, transformer_heads, transformer_dropout)
        
        # self.edge_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim*2), requires_grad=True)

        self.horizon = horizon
        self.embed_dim = embed_dim
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.adjoint = adjoint
        # self.fc = nn.Sequential(nn.Linear(in_features, in_features), nn.ELU(), nn.Linear(in_features, out_features))

    def forward(self, x, eval_times=None):
        # temp = torch.zeros(x.shape[-2:]).type_as(x)
        # temp[:,:self.embed_dim*2] = self.edge_embeddings
        # x = torch.cat([x,temp.unsqueeze(0)])

        if eval_times is None:  
            integration_time = torch.linspace(0, self.horizon, self.horizon+1).float().to(x.device)
        else:
            integration_time = eval_times.type_as(x).to(x.device)

        if self.method == 'dopri5':
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method,
                                options={'max_num_steps': 1000})
        else:
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method)        

        # return out[1:,:-1]
        return out[1:]


class Net(nn.Module):
    def __init__(self, args, edge_index):
        super(Net, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.alpha = args.alpha
        # R-Drop相关参数
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # if args.time_dependence:
            # self.time_control = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1) )
            # self.time_control = nn.Linear(1, 1)

        self.enc = nn.Sequential(nn.Linear(args.input_dim*args.lag, args.hidden_dim),
                                      nn.ReLU())

        self.NADE = ODEBlock(edge_index, args.model_type, args.num_nodes, args.hidden_dim, args.horizon, args.alpha, args.embed_dim, args.transformer_heads, args.transformer_dropout)
        
        self.dec = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(), nn.Linear(args.hidden_dim, args.output_dim))

    def forward(self, X, targets, teacher_forcing_ratio=0.5, apply_r_drop=False):
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        X = X.transpose(1,2).reshape(-1,self.num_node, self.input_dim*self.lag)
        X = self.enc(X)

        nade_out = self.NADE(X)
        nade_out = nade_out.permute(1,0,2,3)

        # 应用Dropout (第一次)
        nade_out_dropped = self.dropout(nade_out)
        # 第一次解码
        out1 = self.dec(nade_out_dropped)

        if apply_r_drop:
            # 第二次应用Dropout (会生成新的随机mask)
            nade_out_dropped_2 = self.dropout(nade_out)
            # 第二次解码
            out2 = self.dec(nade_out_dropped_2)
            return out1, out2  # 训练时返回两个输出
        else:
            return out1  # 评估/测试时返回一个输出

