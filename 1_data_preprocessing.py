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

def extract_features(rows):
    """
    Iterates through raw spreadsheet rows, extracts target categorical values 
    (Party, Region, Position) and numerical/target fields (Year, fat), cleans and 
    normalizes string cases, handles missing data defaults, and parses numeric types.
    """
    parties = []
    regions = []
    positions = []
    years = []
    fats = []
    
    for i in range(1, len(rows)):
        row = rows[i]
        party_val = row[2] if len(row) > 2 and row[2] != '' else 'UNKNOWN'
        region_val = row[3] if len(row) > 3 and row[3] != '' else 'UNKNOWN'
        pos_val = row[6] if len(row) > 6 and row[6] != '' else 'UNKNOWN'
        year_val = row[7] if len(row) > 7 and row[7] != '' else '2004'
        fat_val = row[8] if len(row) > 8 and row[8] != '' else '0'
        
        parties.append(str(party_val).strip().upper())
        regions.append(str(region_val).strip().upper())
        positions.append(str(pos_val).strip().upper())
        
        try:
            years.append(float(year_val))
        except ValueError:
            years.append(2004.0)
            
        try:
            fats.append(int(float(fat_val)))
        except ValueError:
            fats.append(0)
            
    return parties, regions, positions, years, fats
