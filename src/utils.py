import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmsle_score(y_true, y_pred):
    """Calculate RMSLE with handling for zero and negative values"""
    # Add small epsilon to avoid log(0) and ensure positive values
    epsilon = 1e-15
    y_true_log = np.log1p(np.maximum(y_true, epsilon))
    y_pred_log = np.log1p(np.maximum(y_pred, epsilon))
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))

def split_time_series_data(X,y,train_ratio=0.8):
    """
    Split a time series DataFrame into training and validation sets based on a date column.
    
    Parameters:
    - X: DataFrame containing the features.
    - y: Series containing the target variable.
    - train_ratio: Proportion of the data to use for training (default is 0.8).
        
    Returns:
    - X_train: Training features.
    - X_val: Validation features.
    - y_train: Training target variable.
    - y_val: Validation target variable.
    """
    y = y[X.index]  # Align y with sorted X
    
    # Calculate the split index
    split_index = int(len(X) * train_ratio)
    
    # Split the data
    X_train = X.iloc[:split_index]
    X_val = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_val = y.iloc[split_index:]
    
    return X_train, X_val, y_train, y_val