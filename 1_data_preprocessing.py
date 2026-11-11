# Runs the complete data loading and preprocessing pipeline, returning feature matrix X, label array y, and metadata.
import csv
import numpy as np

def load_dataset(file_path):
    """
    Opens and reads a CSV data file row by row using Python's built-in csv module,
    storing all raw text rows into a list and returning them.
    """
    rows = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows