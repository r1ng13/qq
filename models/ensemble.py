import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self, models, output_dim, weights=None):
        """
        Initialize ensemble model
        
        Parameters:
        models (dict): Dictionary of models with names as keys
        output_dim (int): Number of output classes
        weights (dict, optional): Dictionary of model weights, should sum to 1
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleDict(models)
        self.output_dim = output_dim
        
        if weights is None:
            # Default to equal weights
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Normalize weights to sum to 1
            weight_sum = sum(weights.values())
            self.weights = {name: weight / weight_sum for name, weight in weights.items()}
    
    def forward(self, x, H=None, B=None):
        """
        Forward pass through ensemble
        
        Parameters:
        x: Input features
        H: Hypergraph incidence matrix (optional)
        B: Hyperedge weights (optional)
        
        Returns:
        Weighted average of model predictions
        """
        from models.hgnn_plus import HGNNPlus
        from models.dhgcn import DHGCN
        
        outputs = []
        
        for name, model in self.models.items():
            # 检查模型类型
            if isinstance(model, (HGNNPlus, DHGCN)):
                # 对于超图模型，需要正确处理输入
                x_model = x.clone()  # 复制输入防止修改
                
                # 确保输入维度正确
                if len(x_model.shape) != 3:
                    # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                    x_model = x_model.unsqueeze(1)
                
                # 确保H的维度正确
                node_count = x_model.size(1)
                if H is None or H.size(0) != node_count:
                    H_model = torch.ones(node_count, node_count, device=x.device)
                else:
                    H_model = H
                
                # 前向传播
                model_output = model(x_model, H_model)
                
                # 如果输出是[batch_size, num_nodes, output_dim]，需要squeeze
                if len(model_output.shape) == 3:
                    model_output = model_output.squeeze(1)
                
                outputs.append(self.weights[name] * model_output)
            else:
                # 其他模型直接调用
                outputs.append(self.weights[name] * model(x))
        
        # Sum outputs with their respective weights
        ensemble_output = sum(outputs)
        return ensemble_output
    
    def predict_proba(self, x, H=None, B=None):
        """Get probability predictions from the ensemble"""
        with torch.no_grad():
            logits = self.forward(x, H, B)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def get_individual_predictions(self, x, H=None, B=None):
        """Get predictions from individual models in the ensemble"""
        from models.hgnn_plus import HGNNPlus
        from models.dhgcn import DHGCN
        
        individual_preds = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                # 检查模型类型
                if isinstance(model, (HGNNPlus, DHGCN)):
                    # 对于超图模型，需要正确处理输入
                    x_model = x.clone()  # 复制输入防止修改
                    
                    # 确保输入维度正确
                    if len(x_model.shape) != 3:
                        # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                        x_model = x_model.unsqueeze(1)
                    
                    # 确保H的维度正确
                    node_count = x_model.size(1)
                    if H is None or H.size(0) != node_count:
                        H_model = torch.ones(node_count, node_count, device=x.device)
                    else:
                        H_model = H
                    
                    # 前向传播
                    logits = model(x_model, H_model)
                    
                    # 如果输出是[batch_size, num_nodes, output_dim]，需要squeeze
                    if len(logits.shape) == 3:
                        logits = logits.squeeze(1)
                else:
                    # 其他模型直接调用
                    logits = model(x)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                individual_preds[name] = preds
        
        return individual_preds