import numpy as np
import 1_data

def train_test_split_scratch(X, y, test_size=0.2, random_seed=42):
    """
    Splits the feature matrix X and label array y into training and testing sets 
    from scratch using NumPy, shuffling randomly with a fixed seed for reproducibility.
    """
    np.random.seed(random_seed)
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    test_size_count = int(num_samples * test_size)
    test_indices = indices[:test_size_count]
    train_indices = indices[test_size_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test