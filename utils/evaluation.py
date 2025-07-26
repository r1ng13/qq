import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader
    
    Parameters:
    model (torch.nn.Module): The model to evaluate
    data_loader (torch.utils.data.DataLoader): Data loader for evaluation
    device (torch.device): Device to run evaluation on
    
    Returns:
    tuple: (metrics dict, confusion matrix)
    """
    from models.hgnn_plus import HGNNPlus
    from models.dhgcn import DHGCN
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            # 为不同类型的模型正确创建超图
            if isinstance(model, (HGNNPlus, DHGCN)):
                # 对于HGNNPlus和DHGCN，我们需要重新组织特征
                if len(features.shape) != 3:
                    # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                    features = features.unsqueeze(1)
                # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
                node_count = features.size(1)
                H = torch.ones(node_count, node_count, device=device)
                outputs = model(features, H)
            else:
                # 常规模型直接调用
                outputs = model(features)
            
            # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(1)  # [batch_size, output_dim]
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return metrics, conf_matrix

def explain_predictions(model, X, feature_names=None):
    """
    Explain model predictions using SHAP values
    
    Parameters:
    model (torch.nn.Module): The model to explain
    X (numpy.ndarray): Input features
    feature_names (list): List of feature names
    
    Returns:
    numpy.ndarray: SHAP values
    """
    from models.hgnn_plus import HGNNPlus
    from models.dhgcn import DHGCN
    import torch
    
    model.eval()
    
    # 为超图模型创建一个包装类
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            self.is_hypergraph_model = isinstance(model, (HGNNPlus, DHGCN))
            
        def __call__(self, x):
            x_tensor = torch.FloatTensor(x)
            device = next(self.model.parameters()).device
            x_tensor = x_tensor.to(device)
            
            if self.is_hypergraph_model:
                # 对于HGNNPlus和DHGCN，我们需要重新组织特征
                if len(x_tensor.shape) != 3:
                    # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                    x_tensor = x_tensor.unsqueeze(1)
                # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
                node_count = x_tensor.size(1)
                H = torch.ones(node_count, node_count, device=device)
                outputs = self.model(x_tensor, H)
            else:
                outputs = self.model(x_tensor)
            
            # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(1)  # [batch_size, output_dim]
                
            return outputs.detach().cpu().numpy()
    
    wrapped_model = ModelWrapper(model)
    
    # Create a SHAP explainer
    explainer = shap.DeepExplainer(wrapped_model, X[:100])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    return shap_values

def generate_risk_report(model, X, y, df, output_path):
    """
    Generate a risk report for the given model and data
    
    Parameters:
    model (torch.nn.Module): The model to use
    X (numpy.ndarray): Input features
    y (numpy.ndarray): True labels
    df (pandas.DataFrame): Original dataframe
    output_path (str): Path to save the risk report
    """
    from models.hgnn_plus import HGNNPlus
    from models.dhgcn import DHGCN
    
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Get model predictions
    with torch.no_grad():
        # Move input tensor to the same device as the model
        X_tensor = torch.FloatTensor(X).to(device)
        
        # 为不同类型的模型正确创建超图
        if isinstance(model, (HGNNPlus, DHGCN)):
            # 对于HGNNPlus和DHGCN，我们需要重新组织特征
            if len(X_tensor.shape) != 3:
                # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                X_tensor = X_tensor.unsqueeze(1)
            # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
            node_count = X_tensor.size(1)
            H = torch.ones(node_count, node_count, device=device)
            outputs = model(X_tensor, H)
        else:
            # 常规模型直接调用
            outputs = model(X_tensor)
        
        # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
        if len(outputs.shape) == 3:
            outputs = outputs.squeeze(1)  # [batch_size, output_dim]
        
        # Move output back to CPU for numpy operations
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    
    # Create risk report dataframe
    risk_df = pd.DataFrame({
        'package_name': df['package_name'],
        'version': df['version'],
        'true_label': y,
        'predicted_label': preds,
        'malicious_probability': probs[:, 1],
        'benign_probability': probs[:, 0],
        'correctly_classified': y == preds
    })
    
    # Add risk level
    risk_df['risk_level'] = pd.cut(
        risk_df['malicious_probability'], 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Sort by malicious probability in descending order
    risk_df = risk_df.sort_values(by='malicious_probability', ascending=False)
    
    # Save to CSV
    risk_df.to_csv(output_path, index=False)
    
    return risk_df