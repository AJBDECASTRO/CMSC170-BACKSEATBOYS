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

def get_unique_items(items_list):
    """
    Extracts sorted unique category values using set
    """
    unique_set = set()
    for item in items_list:
        unique_set.add(item)
    return sorted(list(unique_set))

def build_mapping(unique_items):
    """
    Constructs an index mapping dictionary that maps each unique category string 
    to a unique integer index for one-hot vector positioning. (In this case, if it belongs in a fat dynasty or not)
    """
    mapping = {}
    for i in range(len(unique_items)):
        mapping[unique_items[i]] = i
    return mapping

def compute_mean_std(values_list):
    """
    Computes the numerical mean and standard deviation of a list of numbers 
    to facilitate z-score.
    """
    total = 0.0
    count = 0.0
    for val in values_list:
        total += val
        count += 1.0
    mean = total / count if count > 0 else 0.0
    
    variance_sum = 0.0
    for val in values_list:
        diff = val - mean
        variance_sum += diff * diff
    variance = variance_sum / count if count > 0 else 1.0
    std = variance ** 0.5
    if std == 0.0:
        std = 1.0
    return mean, std

def encode_features(parties, regions, positions, years):
    """
    Encodes categorical features into sparse one-hot binary vectors, standardizes 
    the numerical year feature via z-score scaling, and compiles everything into 
    a 2D NumPy feature matrix X along with a metadata dictionary.
    """
    unique_parties = get_unique_items(parties)
    unique_regions = get_unique_items(regions)
    unique_positions = get_unique_items(positions)
    
    party_map = build_mapping(unique_parties)
    region_map = build_mapping(unique_regions)
    position_map = build_mapping(unique_positions)
    
    year_mean, year_std = compute_mean_std(years)
    
    encoded_matrix = []
    for i in range(len(parties)):
        p_vec = [0.0] * len(unique_parties)
        if parties[i] in party_map:
            p_vec[party_map[parties[i]]] = 1.0
            
        r_vec = [0.0] * len(unique_regions)
        if regions[i] in region_map:
            r_vec[region_map[regions[i]]] = 1.0
            
        pos_vec = [0.0] * len(unique_positions)
        if positions[i] in position_map:
            pos_vec[position_map[positions[i]]] = 1.0
            
        norm_year = (years[i] - year_mean) / year_std
        
        feature_vector = p_vec + r_vec + pos_vec + [norm_year]
        encoded_matrix.append(feature_vector)
        
    return np.array(encoded_matrix, dtype=float), {
        'party_map': party_map,
        'region_map': region_map,
        'position_map': position_map,
        'year_mean': year_mean,
        'year_std': year_std
    }

def preprocess_pipeline(file_path):
    """
    Executes the end-to-end preprocessing pipeline by sequentially loading data, 
    extracting features, encoding matrices, and returning X, y, and metadata.
    """
    rows = load_dataset(file_path)
    parties, regions, positions, years, fats = extract_features(rows)
    X, metadata = encode_features(parties, regions, positions, years)
    y = np.array(fats, dtype=int)
    return X, y, metadata