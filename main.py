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
