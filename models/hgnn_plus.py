import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6, alpha=0.2):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Define learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * num_heads)))
        self.a_vertices = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_edges = nn.Parameter(torch.zeros(size=(out_features, 1)))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a_vertices.data)
        nn.init.xavier_uniform_(self.a_edges.data)
        
        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x, H):
        """
        x: Node features (B x N x F) or (N x F)
        H: Hypergraph incidence matrix (N x E) or (B x N x E)
        """
        # 检查输入维度
        original_shape = x.shape
        if len(original_shape) == 2:  # (N x F)
            # 添加批次维度
            x = x.unsqueeze(0)  # (1 x N x F)
            
        batch_size, num_nodes, num_features = x.size()
        
        # 检查超图维度
        if len(H.shape) == 2:  # (N x E)
            # 扩展到每个批次
            H = H.unsqueeze(0).expand(batch_size, -1, -1)  # (B x N x E)
        
        # 处理每个批次
        outputs = []
        for i in range(batch_size):
            x_i = x[i]  # (N x F)
            H_i = H[i] if len(H.shape) == 3 else H  # (N x E)
            
            # 线性变换
            Wh_i = torch.matmul(x_i, self.W)  # (N x (out_features * num_heads))
            Wh_i = Wh_i.view(num_nodes, self.num_heads, self.out_features)  # (N x num_heads x out_features)
            
            # 计算顶点的注意力系数
            attn_vertices = torch.matmul(Wh_i, self.a_vertices)  # (N x num_heads x 1)
            attn_vertices = self.leakyrelu(attn_vertices).squeeze(2)  # (N x num_heads)
            
            # 将注意力应用到超边
            attn_edges = F.softmax(torch.matmul(H_i.t(), attn_vertices), dim=0)  # (E x num_heads)
            
            # 通过超图传播节点特征
            head_outputs = []
            for h in range(self.num_heads):
                # 计算加权关联矩阵
                H_weighted = H_i * attn_edges[:, h].unsqueeze(0)  # (N x E)
                
                # 传播特征
                head_output = torch.matmul(H_weighted, torch.matmul(H_i.t(), Wh_i[:, h]))  # (N x out_features)
                head_outputs.append(head_output)
                
            # 组合所有头的输出
            output_i = torch.cat(head_outputs, dim=1)  # (N x (out_features * num_heads))
            outputs.append(output_i)
            
        # 堆叠所有批次的输出
        output = torch.stack(outputs, dim=0)  # (B x N x (out_features * num_heads))
        
        # 如果原始输入没有批次维度，则去除批次维度
        if len(original_shape) == 2:
            output = output.squeeze(0)  # (N x (out_features * num_heads))
            
        return output

class HGNNPlus(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_heads=8, dropout=0.6):
        super(HGNNPlus, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(HypergraphAttentionLayer(input_dim, hidden_dims[0], num_heads))
        
        # Hidden layers with multi-head attention
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                HypergraphAttentionLayer(hidden_dims[i] * num_heads, hidden_dims[i+1], num_heads)
            )
        
        # Output layer
        self.final_layer = nn.Linear(hidden_dims[-1] * num_heads, output_dim)
        
        # 添加缺失的属性
        self.use_attention = False  # 默认不使用注意力机制
        self.attn_weights = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        # 初始化attn_weights
        nn.init.xavier_uniform_(self.attn_weights.data)
        
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, H=None, W=None):
        """
        前向传播函数
        
        Parameters:
        x: 输入特征矩阵 (N x F) 或 (B x N x F)
        H: 超图关联矩阵 (N x E)
        W: 超边权重 (E)
        
        Returns:
        output: 分类logits (B x num_classes)
        """
        # 检查输入维度
        original_shape = x.shape
        is_batch_input = len(original_shape) == 3
        
        if not is_batch_input:
            # 如果输入没有批次维度，添加一个批次维度
            x = x.unsqueeze(0)  # [1, N, F]
        
        # 获取批次大小和节点数量
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # 如果没有提供超图，创建一个简单的超图
        if H is None:
            H = torch.ones(num_nodes, num_nodes, device=x.device)
        else:
            # 确保H的维度正确
            if H.size(0) != num_nodes:
                print(f"Warning: Hypergraph dimension ({H.size(0)}) doesn't match node count ({num_nodes}).")
                H = torch.ones(num_nodes, num_nodes, device=x.device)
        
        # 实施注意力
        if self.use_attention:
            # 计算注意力得分
            attn_scores = F.softmax(torch.matmul(x, self.attn_weights), dim=2)
            
            # 对每个批次应用注意力
            batch_outputs = []
            for i in range(batch_size):
                # 将注意力得分应用到超图
                H_weighted = H * attn_scores[i].mean(dim=0, keepdim=True)
                
                # 计算超图拉普拉斯矩阵
                D_v = torch.sum(H_weighted, dim=1)
                D_v_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_v + 1e-8))
                
                D_e = torch.sum(H_weighted, dim=0)
                D_e_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_e + 1e-8))
                
                # 归一化超图拉普拉斯
                L = D_v_inv_sqrt @ H_weighted @ D_e_inv_sqrt @ H_weighted.t() @ D_v_inv_sqrt
                
                # 当前批次的特征
                x_i = x[i]
                
                # 通过层传播
                for layer in self.layers:
                    x_i = F.relu(layer(x_i, L))
                    x_i = self.dropout_layer(x_i)
                
                # 全局平均池化
                x_i = torch.mean(x_i, dim=0)
                
                # 应用分类器
                output = self.final_layer(x_i)  # 使用self.final_layer代替self.classifier
                batch_outputs.append(output)
            
            # 堆叠所有批次的输出
            logits = torch.stack(batch_outputs, dim=0)
        else:
            # 如果不使用注意力，简化计算
            # 计算归一化超图拉普拉斯矩阵
            D_v = torch.sum(H, dim=1)
            D_v_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_v + 1e-8))
            
            D_e = torch.sum(H, dim=0)
            D_e_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_e + 1e-8))
            
            # 计算归一化的超图拉普拉斯
            L = D_v_inv_sqrt @ H @ D_e_inv_sqrt @ H.t() @ D_v_inv_sqrt
            
            # 通过层传播
            for layer in self.layers:
                x = F.relu(layer(x, L))
                x = self.dropout_layer(x)
            
            # 全局平均池化
            x = torch.mean(x, dim=1)  # [B, F]
            
            # 分类
            logits = self.final_layer(x)  # 使用self.final_layer代替self.classifier
        
        # 确保输出维度正确 - 如果单个样本，确保形状为[1, output_dim]而不是[output_dim]
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
            
        return logits