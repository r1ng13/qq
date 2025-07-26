import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHypergraphConv(nn.Module):
    def __init__(self, in_features, out_features, k=10, bias=True):
        super(DynamicHypergraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k  # Number of nearest neighbors
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_dynamic = nn.Parameter(torch.FloatTensor(in_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_dynamic)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def build_dynamic_hypergraph(self, x):
        """Build dynamic hypergraph based on feature similarity"""
        # Transform features for dynamic graph construction
        features = torch.mm(x, self.weight_dynamic)
        
        # Compute pairwise distances
        dist = torch.cdist(features, features)
        
        # Get indices of k nearest neighbors for each node
        _, indices = torch.topk(dist, k=self.k, dim=1, largest=False)
        
        # Create hypergraph incidence matrix H
        N = x.size(0)
        H = torch.zeros(N, N, device=x.device)
        
        # For each node, create a hyperedge connecting to its k nearest neighbors
        for i in range(N):
            H[i, indices[i]] = 1.0
        
        return H
    
    def forward(self, x, H=None):
        """
        x: Node features (N x in_features)
        H: Optional hypergraph incidence matrix (N x E)
        """
        if H is None:
            # Build dynamic hypergraph
            H = self.build_dynamic_hypergraph(x)
        
        # Calculate node degree matrix D
        D = torch.diag(torch.sum(H, dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(H, dim=1) + 1e-10))
        
        # Calculate hyperedge degree matrix B
        B = torch.diag(torch.sum(H, dim=0))
        B_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(H, dim=0) + 1e-10))
        
        # Hypergraph Laplacian
        L = torch.eye(x.size(0), device=x.device) - D_inv_sqrt @ H @ B_inv_sqrt @ H.t() @ D_inv_sqrt
        
        # Propagate node features through hypergraph
        x = (1 - 0.5) * L @ x + 0.5 * x
        
        # Linear transformation
        x = x @ self.weight
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

class DHGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, k=10, dropout=0.5):
        super(DHGCN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(DynamicHypergraphConv(input_dim, hidden_dims[0], k=k))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(DynamicHypergraphConv(hidden_dims[i], hidden_dims[i+1], k=k))
        
        # Output layer
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # 将dropout从浮点数改为nn.Dropout层
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 保存k参数
        self.k = k
        
    def forward(self, x, H=None):
        """
        前向传播函数
        
        Parameters:
        x: 节点特征矩阵 (N x F) 或 (B x N x F)
        H: 超图关联矩阵 (N x E)
        
        Returns:
        output: 分类logits (B x num_classes)
        """
        # 检查输入维度并处理批次
        original_shape = x.shape
        is_batch_input = len(original_shape) == 3
        
        if not is_batch_input:
            # 如果输入没有批次维度，添加一个批次维度
            x = x.unsqueeze(0)  # [1, N, F]
        
        # 获取批次大小和节点数量
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # 如果没有提供超图，创建一个动态超图
        if H is None:
            # 为每个样本构建动态超图
            batch_outputs = []
            for i in range(batch_size):
                features_i = x[i]
                
                # 计算相似度矩阵
                sim = torch.mm(features_i, features_i.t())
                
                # 获取k近邻
                _, indices = torch.topk(sim, k=min(self.k, num_nodes), dim=1)
                
                # 构建邻接矩阵
                adj = torch.zeros(num_nodes, num_nodes, device=x.device)
                for j in range(num_nodes):
                    adj[j, indices[j]] = 1
                
                # 计算超图拉普拉斯矩阵
                D = torch.sum(adj, dim=1)
                D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-8))
                L = torch.eye(num_nodes, device=x.device) - D_inv_sqrt @ adj @ D_inv_sqrt
                
                # 通过动态GCN层
                h = features_i
                for layer in self.layers:
                    h = F.relu(layer(h, L))
                    h = self.dropout_layer(h)
                
                # 全局池化
                h = torch.mean(h, dim=0)
                
                # 分类
                output = self.final_layer(h)
                batch_outputs.append(output)
            
            # 堆叠所有批次的输出
            logits = torch.stack(batch_outputs, dim=0)
        else:
            # 使用提供的超图
            # 确保H与节点数量匹配
            if H.size(0) != num_nodes:
                print(f"Warning: Hypergraph dimension ({H.size(0)}) doesn't match node count ({num_nodes}).")
                H = torch.ones(num_nodes, num_nodes, device=x.device)
            
            # 计算拉普拉斯矩阵
            D = torch.sum(H, dim=1)
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-8))
            L = torch.eye(num_nodes, device=x.device) - D_inv_sqrt @ H @ D_inv_sqrt
            
            # 为每个批次复制拉普拉斯矩阵
            L_expanded = L.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 处理每个批次
            batch_outputs = []
            for i in range(batch_size):
                h = x[i]
                for layer in self.layers:
                    h = F.relu(layer(h, L))
                    h = self.dropout_layer(h)
                
                # 全局池化
                h = torch.mean(h, dim=0)
                
                # 分类
                output = self.final_layer(h)
                batch_outputs.append(output)
            
            # 堆叠所有批次的输出
            logits = torch.stack(batch_outputs, dim=0)
        
        # 确保输出维度正确 - 如果单个样本且输出是一维的，确保形状为[1, output_dim]而不是[output_dim]
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
            
        return logits