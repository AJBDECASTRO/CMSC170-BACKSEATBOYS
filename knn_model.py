def minkow_distance(x, v, p):
    """
    Calculates the Minkowski distance between two feature vectors x and v 
    raised to power p (p=2 for Euclidean distance).
    """
    d = 0.0
    for i in range(len(x)):
        # Sum absolute differences raised to the p-th power element-wise
        d += abs(x[i] - v[i]) ** p
    d = d ** (1 / p)  # Take the 1/p-th root
    return d

def predict_knn(train_x, train_y, test_point, k, p):
    """
    Predicts the target label for a single test point using k-Nearest Neighbors 
    by computing distances, sorting, and performing a frequency majority vote.
    """
    distances = []
    freq = {}

    # Calculate distance from test point to every training sample
    for i in range(len(train_x)):
        dist = minkow_distance(test_point, train_x[i], p)
        distances.append((dist, train_y[i]))
    
    distances.sort()  # Sort distances in ascending order
    k_nearest = distances[:k]  # Select the k nearest neighbors

    # Count occurrences of each class label among the k nearest neighbors
    for item in k_nearest:
        label = item[1]
        freq[label] = freq.get(label, 0) + 1
    
    # Return the label with the highest frequency count
    max_value = max(freq, key=freq.get)
    return max_value

def run_knn_experiment(X_train, y_train, X_test, y_test, k=5, p=2):
    """
    Executes the full k-NN evaluation loop over the test set, tracking correct 
    predictions and returning overall accuracy and predicted labels.
    """
    # Ensure inputs are standard Python lists for safe iteration
    train_x = X_train.tolist() if hasattr(X_train, 'tolist') else X_train
    train_y = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
    test_x = X_test.tolist() if hasattr(X_test, 'tolist') else X_test
    test_y = y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    
    correct = 0
    predictions = []
    tp = tn = fp = fn = 0
    
    # Iterate through all test samples and evaluate predictions
    for i in range(len(test_x)):
        pred = predict_knn(train_x, train_y, test_x[i], k, p)
        predictions.append(pred)
        if pred == test_y[i] == 1:              # True positive
            tp += 1
        elif pred == test_y[i] == 0:            # True negative
            tn += 1
        elif pred == 1 and test_y[i] == 0:      # False positive
            fp += 1
        else:                                   # False negative
            fn += 1

    print('KNN CONFUSION MATRIX')
    print(f'{tp}\t{fn}\n{fp}\t{tn}\n')
    
    accuracy = ((tp + tn) / len(test_x)) * 100
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    f1 = 2 * ((precision * recall) / (precision + recall))
    return (accuracy, precision, recall, f1), predictions