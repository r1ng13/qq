import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def extract_text_features(df, text_columns, max_features=1000):
    """
    Extract features from text columns using TF-IDF
    
    Parameters:
    df (pandas.DataFrame): Dataset
    text_columns (list): List of text column names
    max_features (int): Maximum number of features to extract
    
    Returns:
    numpy.ndarray: Text feature matrix
    """
    # Combine text columns
    text_data = df[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    
    # Extract TF-IDF features
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    text_features = vectorizer.fit_transform(text_data).toarray()
    
    return text_features

def extract_api_features(df):
    """
    Extract features from API calls
    
    Parameters:
    df (pandas.DataFrame): Dataset
    
    Returns:
    numpy.ndarray: API feature matrix
    """
    # Get all unique API calls
    all_apis = set()
    for apis in df['api_calls']:
        if isinstance(apis, list):
            all_apis.update(apis)
    
    # Create API feature matrix
    api_features = np.zeros((len(df), len(all_apis)))
    
    # Map API calls to indices
    api_to_idx = {api: i for i, api in enumerate(all_apis)}
    
    # Populate API feature matrix
    for i, apis in enumerate(df['api_calls']):
        if isinstance(apis, list):
            for api in apis:
                j = api_to_idx.get(api)
                if j is not None:
                    api_features[i, j] = 1.0
    
    return api_features

def extract_metadata_features(df):
    """
    Extract features from package metadata
    
    Parameters:
    df (pandas.DataFrame): Dataset
    
    Returns:
    numpy.ndarray: Metadata feature matrix
    """
    # Extract numeric features
    numeric_features = df[['total_file', 'package_size', 'num_api_calls', 
                          'has_obfuscated_name']].values
    
    # Include danger API indicators
    danger_columns = [col for col in df.columns if col.startswith('has_')]
    danger_features = df[danger_columns].values
    
    # Combine features
    metadata_features = np.hstack([numeric_features, danger_features])
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(metadata_features)
    
    return scaled_features

def extract_features(df):
    """
    Extract features from the dataset
    
    Parameters:
    df (pandas.DataFrame): Dataset
    
    Returns:
    tuple: (features, labels)
    """
    # Extract features from text, API calls, and metadata
    text_features = extract_text_features(df, ['package_name'], max_features=500)
    api_features = extract_api_features(df)
    metadata_features = extract_metadata_features(df)
    
    # Combine all features
    features = np.hstack([text_features, api_features, metadata_features])
    
    # Get labels
    labels = df['malicious'].values
    
    return features, labels