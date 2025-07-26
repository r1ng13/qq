import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import os

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Parameters:
    train_losses (list): Training losses
    val_losses (list): Validation losses
    train_accs (list): Training accuracies
    val_accs (list): Validation accuracies
    save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot a confusion matrix
    
    Parameters:
    conf_matrix (numpy.ndarray): Confusion matrix to plot
    save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Benign', 'Malicious'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_importances, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Parameters:
    feature_importances (numpy.ndarray): Feature importance values
    feature_names (list): Names of features
    top_n (int): Number of top features to show
    save_path (str): Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(feature_importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, auc, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    fpr (numpy.ndarray): False positive rates
    tpr (numpy.ndarray): True positive rates
    auc (float): Area under the ROC curve
    save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()