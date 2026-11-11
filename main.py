# main.py
import numpy as np
from data_preprocessing import preprocess_pipeline
from train_test_split import train_test_split_scratch  # Assumes your split function is saved in train_test_split.py

# 1. Execute pipeline with raw lists returned for comparison printing
print("Loading dataset and running preprocessing pipeline...")
X, y, metadata, parties, regions, positions, years = preprocess_pipeline('POLDYNASTY_ATENEO_DATASET-Data.csv')

# 2. Print side-by-side comparison for the first 5 rows
print("\n" + "=" * 120)
print(f"{'#':<3} | {'RAW STARTING DATA':<52} | {'PRE-PROCESSED NUMERICAL DATA (X & y)'}")
print("-" * 120)
for i in range(5):
    raw_str = f"P:{parties[i][:10]} | R:{regions[i][:12]} | Pos:{positions[i][:8]} | Yr:{int(years[i])}"
    active_indices = np.nonzero(X[i])[0].tolist()
    proc_str = f"y={y[i]} | Yr(norm)={X[i][-1]:.2f} | Active Indices={active_indices}"
    print(f"{i+1:<3} | {raw_str:<52} | {proc_str}")
print("=" * 120 + "\n")

# 3. Perform Train/Test Split
print("Performing 80/20 Train/Test split...")
X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, random_seed=42)

# 4. Print Split Statistics Table
print("=" * 60)
print(f"{'Dataset Split':<15} | {'Total Rows':<12} | {'Non-Dynasty (0)':<18} | {'Fat Dynasty (1)'}")
print("-" * 60)

# Count zeros and ones in y_train and y_test
train_zeros = int(np.sum(y_train == 0))
train_ones = int(np.sum(y_train == 1))
test_zeros = int(np.sum(y_test == 0))
test_ones = int(np.sum(y_test == 1))

print(f"{'Training Set':<15} | {X_train.shape[0]:<12} | {train_zeros:<18} | {train_ones}")
print(f"{'Testing Set':<15} | {X_test.shape[0]:<12} | {test_zeros:<18} | {test_ones}")
print("=" * 60)

# 5. Print percentage breakdown to verify ratio preservation
train_ratio_ones = (train_ones / len(y_train)) * 100
test_ratio_ones = (test_ones / len(y_test)) * 100
print(f"Fat Dynasty % in Training: {train_ratio_ones:.2f}%")
print(f"Fat Dynasty % in Testing:  {test_ratio_ones:.2f}%")