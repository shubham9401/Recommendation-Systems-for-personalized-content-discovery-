import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def parse_combined_file(filepath):
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Parsing {filepath} ({file_size_mb:.2f} MB)...")
    
    data = []
    current_movie = None
    
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc=os.path.basename(filepath)):
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                user_id, rating, date_str = line.split(',')
                data.append((int(user_id), current_movie, int(rating), date_str))
                
    df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating', 'date'])
    df['rating'] = df['rating'].astype(np.int8)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Parsed {len(df)} rows from {filepath}")
    return df

def load_all_ratings(data_dir=config.RAW_DATA_DIR, sample_frac=1.0):
    dfs = []
    for i in range(1, 5):
        filepath = os.path.join(data_dir, f"combined_data_{i}.txt")
        if os.path.exists(filepath):
            df = parse_combined_file(filepath)
            if sample_frac < 1.0:
                print(f"Sampling {sample_frac*100}% of users to save memory...")
                uids = df['user_id'].unique()
                rng = np.random.default_rng(config.RANDOM_SEED + i)
                sel = rng.choice(uids, size=max(1, int(len(uids) * sample_frac)), replace=False)
                df = df[df['user_id'].isin(sel)].copy()
            dfs.append(df)
            import gc
            gc.collect()
        else:
            print(f"File not found: {filepath}")
            
    if not dfs:
        print("No rating files to load.")
        return pd.DataFrame()
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    mem_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Total shape: {combined_df.shape}")
    print(f"Memory usage: {mem_mb:.2f} MB")
    
    return combined_df

def load_movie_titles(filepath):
    print(f"Loading movie titles from {filepath}...")
    records = []
    
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                m_id, year_str, title = parts
                year = float(year_str) if year_str != 'NULL' else np.nan
                records.append((int(m_id), year, title))
                
    df = pd.DataFrame(records, columns=['movie_id', 'year', 'title'])
    print(f"Loaded {len(df)} movie titles")
    return df

def save_processed(df, path):
    df.to_parquet(path, engine='pyarrow', index=False)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved processed data to {path} ({file_size_mb:.2f} MB)")

def load_processed(path):
    df = pd.read_parquet(path)
    print(f"Loaded {df.shape} from {path}")
    return df

if __name__ == "__main__":
    # TODO: add arg parsing later for different data splits
    ratings_df = load_all_ratings()
    
    if not ratings_df.empty:
        processed_path = os.path.join(config.PROCESSED_DATA_DIR, "all_ratings.parquet")
        save_processed(ratings_df, processed_path)
        
    titles_path = os.path.join(config.RAW_DATA_DIR, "movie_titles.txt")
    if os.path.exists(titles_path):
        titles_df = load_movie_titles(titles_path)
