def train_perceptron(X_train, y_train, learning_rate=0.01, bias_weight=0.1, max_iterations=5):
    """
    Trains a bipolar Perceptron model using iterative weight adjustments 
    based on misclassified samples up to max_iterations.
    """
    dataset = []
    for i in range(len(X_train)):
        # Prepend bias term x0 = 1.0 to feature vector
        x_values = [1.0] + list(X_train[i])
        # Map binary target classes (0 -> -1.0, 1 -> 1.0) for bipolar thresholding
        target_d = 1.0 if y_train[i] == 1 else -1.0
        dataset.append((x_values, target_d))

    num_features = len(X_train[0])
    weights = [bias_weight] + [0.0] * num_features
    threshold = 0.0
    
    iteration = 0
    converged = False

    # Iteratively update weights until convergence or max iterations reached
    while not converged and iteration < max_iterations:
        iteration += 1
        converged = True

        for x_values, target_d in dataset:
            # Compute linear combination v = sum(x_i * w_i)
            v = 0.0
            for i in range(len(x_values)):
                v += x_values[i] * weights[i]
            
            # Apply threshold function for bipolar output (-1.0 or 1.0)
            y = 1.0 if v >= threshold else -1.0

            # If prediction does not match target, adjust weights: w = w + r * x * (d - y)
            if y != target_d:
                converged = False
                for i in range(len(weights)):
                    weights[i] += learning_rate * x_values[i] * (target_d - y)
                    
        if converged:
            break

    return weights

def predict_perceptron(weights, test_point):
    """
    Predicts the class label for a single test point using trained Perceptron weights,
    mapping the bipolar output back to standard binary space (0 or 1).
    """
    x_values = [1.0] + list(test_point)
    v = 0.0
    for i in range(len(x_values)):
        v += x_values[i] * weights[i]
    
    # Map bipolar result back to 0 or 1
    return 1 if v >= 0.0 else 0

def run_perceptron_experiment(X_train, y_train, X_test, y_test, lr=0.01, bias=0.1, max_iter=5):
    """
    Trains the Perceptron model on training data and evaluates it against the test set, 
    returning test accuracy, final weights, and prediction outputs.
    """
    weights = train_perceptron(X_train, y_train, learning_rate=lr, bias_weight=bias, max_iterations=max_iter)
    
    correct = 0
    predictions = []
    
    tp = tn = fp = fn = 0
    
    # Evaluate model predictions against actual test labels
    for i in range(len(X_test)):
        pred = predict_perceptron(weights, X_test[i])
        predictions.append(pred)
        if pred == y_test[i] == 1:              # True positive
            tp += 1
        elif pred == y_test[i] == 0:            # True negative
            tn += 1
        elif pred == 1 and y_test[i] == 0:      # False positive
            fp += 1
        else:                                   # False negative
            fn += 1

    accuracy = ((tp + tn) / len(X_test)) * 100
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    f1 = 2 * ((precision * recall) / (precision + recall))
    return (accuracy, precision, recall, f1), weights, predictions

    # accuracy = (correct / len(X_test)) * 100
    # return accuracy, weights, predictions