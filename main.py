# main.py
from data_preprocessing import preprocess_pipeline
from knn_model import run_knn_experiment
from perceptron_model import run_perceptron_experiment
from naive_bayes_model import run_naive_bayes_experiment

print("=" * 60)
print("LOADING AND PREPROCESSING DATASET...")
print("=" * 60)

# 1. Load dataset and execute the preprocessing pipeline
X, y, metadata, parties, regions, positions, years = preprocess_pipeline('POLDYNASTY_ATENEO_DATASET-Data.csv')

sample_limit = 10000  
X = X[:sample_limit]
y = y[:sample_limit]
# ------------------------

# 2. Perform a direct sequential split (80% train, 20% test)
dataset_size = X.shape[0]
train_size = round(dataset_size * 0.80)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Total Dataset Rows: {dataset_size}")
print(f"Training Rows:      {train_size}")
print(f"Testing Rows:       {dataset_size - train_size}\n")

# --- 3. Run k-Nearest Neighbors (k-NN) Experiment ---
print("=" * 60)
print("EXECUTING k-NEAREST NEIGHBORS (k=5, p=2)...")
print("=" * 60)
knn_metrics, knn_preds = run_knn_experiment(X_train, y_train, X_test, y_test, k=5, p=2)
print(f"k-NN Test Accuracy: {knn_metrics[0]:.2f}%")
print(f"Sample k-NN Predictions (first 5): {knn_preds[:5]}")
print()
print(f"k-NN Test Precision: {knn_metrics[1]:.2f}%")
print(f"Sample k-NN Predictions (first 5): {knn_preds[:5]}")
print()
print(f"k-NN Test Recall: {knn_metrics[2]:.2f}%")
print(f"Sample k-NN Predictions (first 5): {knn_preds[:5]}")
print()
print(f"k-NN Test F1-score: {knn_metrics[3]:.2f}%")
print(f"Sample k-NN Predictions (first 5): {knn_preds[:5]}")
print()

# --- 4. Run Perceptron Experiment ---
print("=" * 60)
print("EXECUTING PERCEPTRON (lr=0.01, bias=0.1)...")
print("=" * 60)
perceptron_metrics, final_weights, perceptron_preds = run_perceptron_experiment(
    X_train, y_train, X_test, y_test, lr=0.01, bias=0.1, max_iter=5
)
print(f"Perceptron Test Accuracy: {perceptron_metrics[0]:.2f}%")
print(f"Sample Perceptron Predictions (first 5): {perceptron_preds[:5]}")
print()
print(f"Perceptron Test Precision: {perceptron_metrics[1]:.2f}%")
print(f"Sample Perceptron Predictions (first 5): {perceptron_preds[:5]}")
print()
print(f"Perceptron Test Recall: {perceptron_metrics[2]:.2f}%")
print(f"Sample Perceptron Predictions (first 5): {perceptron_preds[:5]}")
print()
print(f"Perceptron Test F1-score: {perceptron_metrics[3]:.2f}%")
print(f"Sample Perceptron Predictions (first 5): {perceptron_preds[:5]}")
print()


# --- 5. Run Naive Bayes Experiment ---
print("=" * 60)
print("EXECUTING NAIVE BAYES (Laplace smoothing k=1.0)...")
print("=" * 60)
nb_acc, nb_preds = run_naive_bayes_experiment(X_train, y_train, X_test, y_test, k=1.0)
print(f"Naive Bayes Test Accuracy: {nb_acc:.2f}%")
print(f"Sample Naive Bayes Predictions (first 5): {nb_preds[:5]}")
print("=" * 60)