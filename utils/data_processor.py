import pandas as pd
import numpy as np
import re
import json
from ast import literal_eval

def load_data(data_path):
    """
    Load the PypiGuard dataset
    
    Parameters:
    data_path (str): Path to the dataset
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset for analysis
    
    Parameters:
    df (pandas.DataFrame): Raw dataset
    
    Returns:
    pandas.DataFrame: Preprocessed dataset
    """
    # Create a binary label for malicious/benign
    df['malicious'] = (df['threat_type'] == 'malicious').astype(int)
    
    # Convert api_calls from string to list and handle NaN values properly
    df['api_calls'] = df['api_calls'].apply(
        lambda x: literal_eval(x) if isinstance(x, str) and x != 'NaN' and pd.notna(x) else []
    )
    
    # Fill NaN values
    df['function'] = df['function'].fillna('')
    
    # Extract additional features
    df['num_api_calls'] = df['api_calls'].apply(len)
    df['has_obfuscated_name'] = df['function'].apply(
        lambda x: 1 if re.search(r'[0O1lI]{5,}', str(x)) else 0
    )
    
    # Dangerous API calls
    dangerous_apis = ['eval', 'exec', 'compile', 'base64', 'subprocess', 'os.system', 
                     'pickle', 'marshal', 'urllib', 'requests', 'socket', 'setInterval']
    
    for api in dangerous_apis:
        df[f'has_{api}'] = df['api_calls'].apply(
            lambda x: 1 if any(api in str(call).lower() for call in x) else 0
        )
    
    return df