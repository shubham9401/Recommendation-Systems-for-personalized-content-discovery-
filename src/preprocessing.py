import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from surprise import Reader, Dataset
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def sample_data(df, frac=config.SAMPLE_FRACTION, min_user_ratings=20, min_movie_ratings=50):
    print("Filtering and sampling data...")
    original_size = len(df)
    
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    filtered_df = df[df['user_id'].isin(valid_users)]
    
    movie_counts = filtered_df['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    filtered_df = filtered_df[filtered_df['movie_id'].isin(valid_movies)]
    
    size_after_filtering = len(filtered_df)
    
    unique_users = filtered_df['user_id'].unique()
    sampled_users = pd.Series(unique_users).sample(frac=frac, random_state=config.RANDOM_SEED)
    sampled_df = filtered_df[filtered_df['user_id'].isin(sampled_users)]
    
    size_after_sampling = len(sampled_df)
    
    num_users = sampled_df['user_id'].nunique()
    num_movies = sampled_df['movie_id'].nunique()
    sparsity = 1.0 - (size_after_sampling / (num_users * num_movies)) if num_users * num_movies > 0 else 0.0
    
    print(f"Original size: {original_size}")
    print(f"After filtering: {size_after_filtering}")
    print(f"After sampling: {size_after_sampling}")
    print(f"Sparsity: {sparsity * 100:.4f}%")
    
    return sampled_df

def create_train_test_split(df, test_size=config.TEST_SIZE, strategy='temporal'):
    if strategy == 'temporal':
        print(f"Creating temporal train-test split (test_size={test_size})...")
        df_sorted = df.sort_values(['user_id', 'date'])
        
        # TODO: try ALS as alternative to this vectorized approach if memory becomes an issue
        df_sorted['rank'] = df_sorted.groupby('user_id')['date'].rank(method='first')
        df_sorted['count'] = df_sorted.groupby('user_id')['user_id'].transform('count')
        
        train_mask = df_sorted['rank'] <= (df_sorted['count'] * (1 - test_size))
        train_df = df_sorted[train_mask].drop(columns=['rank', 'count'])
        test_df = df_sorted[~train_mask].drop(columns=['rank', 'count'])
        
    elif strategy == 'random':
        print(f"Creating random train-test split (test_size={test_size})...")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=config.RANDOM_SEED, shuffle=True)
    else:
        raise ValueError("Strategy must be 'temporal' or 'random'")
        
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    train_users = set(train_df['user_id'])
    test_users = set(test_df['user_id'])
    overlap = len(train_users.intersection(test_users))
    print(f"Users in both sets: {overlap}")
    
    return train_df, test_df

def build_surprise_dataset(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    return data

def build_sparse_matrix(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_idx = user_encoder.fit_transform(df['user_id'])
    item_idx = item_encoder.fit_transform(df['movie_id'])
    
    matrix = csr_matrix((df['rating'], (user_idx, item_idx)), 
                        shape=(len(user_encoder.classes_), len(item_encoder.classes_)))
    
    return matrix, user_encoder, item_encoder
