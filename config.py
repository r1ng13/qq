import os

# Data settings
DATA_PATH = "d:/trae项目/实验v3/pypiguard dataset.csv"
OUTPUT_DIR = "output"

# Feature engineering settings
TEXT_FEATURES = ['package_name', 'author', 'author_email', 'summary', 'description']
MAX_VOCAB_SIZE = 5000
MAX_TEXT_LENGTH = 100

# API call categories
API_CATEGORIES = [
    'file_operations', 'network_operations', 'system_operations',
    'crypto_operations', 'process_operations', 'registry_operations'
]

# Hypergraph construction settings
KNN_K = 5
NUM_HYPEREDGES = 100
HYPEREDGE_WEIGHT_THRESHOLD = 0.1

# Model parameters
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.5
NUM_HEADS = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
EPOCHS = 100

# Hypergraph convolution parameters
HGC_BIAS = True
USE_ATTENTION = True
ATTENTION_HEADS = 8

# Ensemble parameters
ENSEMBLE_WEIGHTS = {
    'hgnn_plus': 0.5,
    'dhgcn': 0.5
}

# Evaluation settings
THRESHOLD = 0.5
TOP_K_FEATURES = 10

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 100
CMAP = 'viridis'
HYPERGRAPH_NODE_SIZE = 50
HYPERGRAPH_EDGE_WIDTH = 2