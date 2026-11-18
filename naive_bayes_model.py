import math

def discretize_value(value):
    """
    Converts a continuous feature value into a discrete token by rounding a after converting to a float value)
    """
    return round(float(value), 2)

def train_naive_bayes(X_train, y_train, k=1.0):
    """
    Trains a Naive Bayes classifier using the additive smoothing formula
    P(w | class) = (count(w in class) + k) / (count(# of words in class) + k * dictionarySize).
    """
    train_x = X_train.tolist() if hasattr(X_train, 'tolist') else X_train
    train_y = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
    num_features = len(train_x[0])

    class_counts = {}
    feature_value_counts = {}
    feature_totals = {}
    # dictionarySize: number of unique values found on both classes for each feature
    vocabularies = [set() for _ in range(num_features)]

    for i in range(len(train_x)):
        label = train_y[i]
        if label not in class_counts:
            class_counts[label] = 0
            feature_value_counts[label] = [{} for _ in range(num_features)]
            feature_totals[label] = [0] * num_features

        class_counts[label] += 1
        for j in range(num_features):
            token = discretize_value(train_x[i][j])
            vocabularies[j].add(token)
            counts = feature_value_counts[label][j]
            counts[token] = counts.get(token, 0) + 1
            feature_totals[label][j] += 1

    total_samples = len(train_x)
    classes = sorted(class_counts.keys())
    priors = {}
    for label in classes:
        priors[label] = class_counts[label] / total_samples

    return {
        'classes': classes,
        'priors': priors,
        'feature_value_counts': feature_value_counts,
        'feature_totals': feature_totals,
        'dictionary_sizes': [len(vocab) for vocab in vocabularies],
        'k': k
    }

def compute_log_posteriors(model, test_point):
    """
    Computes the unnormalized log posterior for each class by summing the log
    likelihoods of every feature given the class and adding the log class prior.
    Working in log space avoids underflow when many probabilities are multiplied.
    """
    k = model['k']
    scores = {}

    for label in model['classes']:
        score = math.log(model['priors'][label])
        for j in range(len(test_point)):
            token = discretize_value(test_point[j])
            count = model['feature_value_counts'][label][j].get(token, 0)
            numerator = count + k
            denominator = model['feature_totals'][label][j] + (k * model['dictionary_sizes'][j])
            score += math.log(numerator / denominator)
        scores[label] = score

    return scores

def predict_naive_bayes(model, test_point):
    """
    Predicts the class label for a single test point by selecting the class with
    the highest score.
    """
    scores = compute_log_posteriors(model, test_point)
    best_label = None
    best_score = None

    for label in model['classes']:
        if best_score is None or scores[label] > best_score:
            best_score = scores[label]
            best_label = label

    return best_label

def run_naive_bayes_experiment(X_train, y_train, X_test, y_test, k=1.0):
    """
    Trains the Naive Bayes model on the training data and evaluates it against
    the test set, returning test accuracy and prediction outputs.
    """
    test_x = X_test.tolist() if hasattr(X_test, 'tolist') else X_test
    test_y = y_test.tolist() if hasattr(y_test, 'tolist') else y_test

    model = train_naive_bayes(X_train, y_train, k=k)

    correct = 0
    predictions = []

    for i in range(len(test_x)):
        pred = predict_naive_bayes(model, test_x[i])
        predictions.append(pred)
        if pred == test_y[i]:
            correct += 1

    accuracy = (correct / len(test_x)) * 100
    return accuracy, predictions
