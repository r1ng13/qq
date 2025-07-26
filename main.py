import sys
import os
import io

# Add the directory of this file to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 创建输出目录
model_dir = os.path.join(current_dir, "output")
backup_dir = os.path.join(current_dir, "model_backup")
if not os.path.exists(model_dir):
    try:
        os.mkdir(model_dir)
        print(f"创建输出目录成功: {model_dir}")
    except Exception as e:
        print(f"创建输出目录失败: {e}")

if not os.path.exists(backup_dir):
    try:
        os.mkdir(backup_dir)
        print(f"创建备份目录成功: {backup_dir}")
    except Exception as e:
        print(f"创建备份目录失败: {e}")

import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.hgnn_plus import HGNNPlus
from models.dhgcn import DHGCN
from models.ensemble import EnsembleModel
from utils.data_processor import load_data, preprocess_data
from utils.hypergraph_builder import build_hypergraph
from utils.feature_engineering import extract_features
from utils.evaluation import evaluate_model, generate_risk_report
from visualization.results_visualization import plot_training_curves, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from visualization.hypergraph_visualization import visualize_hypergraph

import config

def train_model(model, train_loader, val_loader, device, epochs=100, output_dir=None):
    """Train the model with the given data loaders"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct = 0, 0
        for features, labels in train_loader:
            batch_size = features.size(0)
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 修改创建超图的方式
            if isinstance(model, (HGNNPlus, DHGCN)):
                # 对于HGNNPlus和DHGCN，我们需要重新组织特征
                if len(features.shape) != 3:
                    # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                    features = features.unsqueeze(1)
                # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
                node_count = features.size(1)
                H = torch.ones(node_count, node_count, device=device)
            else:
                # 为其他模型创建一个简单的超图
                H = torch.ones(features.size(0), features.size(0), device=device)
            
            # 调用模型前向传播
            outputs = model(features, H) if isinstance(model, (HGNNPlus, DHGCN)) else model(features)
            
            # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(1)  # [batch_size, output_dim]
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                # 修改创建超图的方式
                if isinstance(model, (HGNNPlus, DHGCN)):
                    # 对于HGNNPlus和DHGCN，我们需要重新组织特征
                    if len(features.shape) != 3:
                        # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                        features = features.unsqueeze(1)
                    # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
                    node_count = features.size(1)
                    H = torch.ones(node_count, node_count, device=device)
                else:
                    # 为其他模型创建一个简单的超图
                    H = torch.ones(features.size(0), features.size(0), device=device)
                
                # 调用模型前向传播
                outputs = model(features, H) if isinstance(model, (HGNNPlus, DHGCN)) else model(features)
                
                # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
                if len(outputs.shape) == 3:
                    outputs = outputs.squeeze(1)  # [batch_size, output_dim]
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # 使用BytesIO保存到内存中，然后写入文件
            try:
                model_basename = f"{model.__class__.__name__}_best.pth"
                model_path = os.path.join(current_dir, model_basename)
                print(f"Saving model to: {model_path}")
                
                # 首先保存到内存流
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                
                # 然后从内存流写入文件
                with open(model_path, 'wb') as f:
                    f.write(buffer.getvalue())
                
                print(f"Model saved successfully to: {model_path}")
            except Exception as e:
                print(f"保存模型时出错: {e}")
    
    return model, train_losses, val_losses, train_accs, val_accs

def main():
    # 确保 np 可以在函数内部访问
    global np
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OptimalHyperGNN for Malicious Package Detection')
    parser.add_argument('--model', type=str, default='ensemble', choices=['hgnn_plus', 'dhgcn', 'ensemble'],
                        help='Model to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hypergraph_method', type=str, default='multi', 
                        choices=['api', 'knn', 'multi'], help='Method to construct hypergraph')
    parser.add_argument('--visualize', action='store_true', help='Visualize results and hypergraph')
    args = parser.parse_args()
    
    # Determine the absolute path for the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用脚本所在目录作为输出目录
    output_dir = script_dir
    print(f"Output directory set to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(config.DATA_PATH)
    df = preprocess_data(df)
    
    # Extract features
    print("Extracting features...")
    features, labels = extract_features(df)
    
    # Build hypergraph
    print(f"Building hypergraph using {args.hypergraph_method} method...")
    hypergraph = build_hypergraph(df, features, method=args.hypergraph_method)
    
    if args.visualize:
        try:
            visualize_path = os.path.join(output_dir, f"hypergraph_{args.hypergraph_method}.png")
            visualize_hypergraph(hypergraph, visualize_path)
            print(f"Visualization saved to: {visualize_path}")
        except Exception as e:
            print(f"可视化保存失败: {e}")
    
    # Store original indices before splitting
    indices = np.arange(len(features))
    
    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features, labels, indices, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train, y_train, idx_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model based on argument
    input_dim = features.shape[1]
    hidden_dims = config.HIDDEN_DIMS
    output_dim = len(np.unique(labels))
    
    if args.model == 'hgnn_plus':
        model = HGNNPlus(input_dim, hidden_dims, output_dim, num_heads=config.NUM_HEADS).to(device)
    elif args.model == 'dhgcn':
        model = DHGCN(input_dim, hidden_dims, output_dim, dropout=config.DROPOUT).to(device)
    elif args.model == 'ensemble':
        models = {
            'hgnn_plus': HGNNPlus(input_dim, hidden_dims, output_dim, num_heads=config.NUM_HEADS).to(device),
            'dhgcn': DHGCN(input_dim, hidden_dims, output_dim, dropout=config.DROPOUT).to(device)
        }
        
        # Train individual models
        for name, model_instance in models.items():
            print(f"\nTraining {name}...")
            model_instance, train_losses, val_losses, train_accs, val_accs = train_model(
                model_instance, train_loader, val_loader, device, epochs=args.epochs, output_dir=output_dir
            )
            
            if args.visualize:
                try:
                    plot_path = os.path.join(output_dir, f"{name}_training_curves.png")
                    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
                    print(f"Training curves saved to: {plot_path}")
                except Exception as e:
                    print(f"训练曲线保存失败: {e}")
        
        # Combine models into ensemble
        model = EnsembleModel(models, output_dim, config.ENSEMBLE_WEIGHTS).to(device)
    
    # Train model if not ensemble (ensemble models are trained individually above)
    if args.model != 'ensemble':
        print(f"\nTraining {args.model}...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, device, epochs=args.epochs, output_dir=output_dir
        )
        
        if args.visualize:
            try:
                plot_path = os.path.join(output_dir, f"{args.model}_training_curves.png")
                plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
                print(f"Training curves saved to: {plot_path}")
            except Exception as e:
                print(f"训练曲线保存失败: {e}")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics, confusion = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    
    if args.visualize:
        try:
            confusion_path = os.path.join(output_dir, f"{args.model}_confusion_matrix.png")
            plot_confusion_matrix(confusion, confusion_path)
            print(f"Confusion matrix saved to: {confusion_path}")
        except Exception as e:
            print(f"混淆矩阵保存失败: {e}")
        
        # 添加 ROC 曲线可视化
        print("\nGenerating ROC curve...")
        from sklearn.metrics import roc_curve, auc
        
        # 获取测试集的预测概率
        y_true = []
        y_score = []
        
        model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
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
                
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # 获取正类概率
                
                y_true.extend(labels.cpu().numpy())
                y_score.extend(probs.cpu().numpy())
        
        # 计算 ROC 曲线数据
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # 绘制 ROC 曲线
        try:
            roc_path = os.path.join(output_dir, f"{args.model}_roc_curve.png")
            plot_roc_curve(fpr, tpr, roc_auc, roc_path)
            print(f"ROC curve saved to: {roc_path}")
        except Exception as e:
            print(f"ROC曲线保存失败: {e}")
        
        # 添加特征重要性可视化（如果可用）
        try:
            print("\nGenerating feature importance visualization...")
            
            # 对于简单模型，我们可以使用排列重要性方法
            from sklearn.inspection import permutation_importance
            import numpy as np
            
            # 准备用于特征重要性计算的数据
            X_test_np = X_test
            y_test_np = y_test
            
            # 创建一个包装模型以适应 scikit-learn API
            class ModelWrapper:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device
                
                def fit(self, X, y=None):
                    # 空实现，因为模型已经训练好了
                    return self
                
                def predict(self, X):
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        
                        # 为不同类型的模型正确创建超图
                        if isinstance(self.model, (HGNNPlus, DHGCN)):
                            # 对于HGNNPlus和DHGCN，我们需要重新组织特征
                            if len(X_tensor.shape) != 3:
                                # 将特征变为[batch_size, 1, feature_dim]，每个样本只有一个节点
                                X_tensor = X_tensor.unsqueeze(1)
                            # 为每个样本创建超图，大小为[node_count, node_count]，这里node_count为1
                            node_count = X_tensor.size(1)
                            H = torch.ones(node_count, node_count, device=self.device)
                            outputs = self.model(X_tensor, H)
                        else:
                            # 常规模型直接调用
                            outputs = self.model(X_tensor)
                        
                        # 如果输出是[batch_size, num_nodes, output_dim]，我们需要squeeze它
                        if len(outputs.shape) == 3:
                            outputs = outputs.squeeze(1)  # [batch_size, output_dim]
                        
                        _, preds = torch.max(outputs, 1)
                        return preds.cpu().numpy()
            
            wrapped_model = ModelWrapper(model, device)
            
            # 计算排列重要性
            r = permutation_importance(wrapped_model, X_test_np, y_test_np, 
                                      n_repeats=5, random_state=42)
            
            # 创建特征名称（根据实际特征调整）
            feature_names = [f"Feature_{i}" for i in range(X_test_np.shape[1])]
            
            # 绘制特征重要性
            importances = r.importances_mean
            try:
                importance_path = os.path.join(output_dir, f"{args.model}_feature_importance.png")
                plot_feature_importance(importances, feature_names, top_n=20, save_path=importance_path)
                print(f"Feature importance saved to: {importance_path}")
            except Exception as e:
                print(f"特征重要性保存失败: {e}")
        except Exception as e:
            print(f"无法生成特征重要性可视化: {e}")
    
    # Generate risk report
    print("\nGenerating risk report...")
    try:
        report_path = os.path.join(output_dir, f"{args.model}_risk_report.csv")
        generate_risk_report(model, X_test, y_test, df.iloc[idx_test], report_path)
        print(f"Risk report saved to: {report_path}")
    except Exception as e:
        print(f"风险报告保存失败: {e}")
        # 尝试使用另一种方式保存
        try:
            report_path = os.path.join(current_dir, f"{args.model}_risk_report.csv")
            generate_risk_report(model, X_test, y_test, df.iloc[idx_test], report_path)
            print(f"Risk report saved to alternate location: {report_path}")
        except Exception as e2:
            print(f"备用保存也失败: {e2}")

if __name__ == "__main__":
    main()