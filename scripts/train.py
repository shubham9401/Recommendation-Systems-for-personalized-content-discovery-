import os
import sys
import argparse
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_processed, load_all_ratings, save_processed
from src.preprocessing import sample_data, create_train_test_split, build_surprise_dataset
from src.models.knn_cf import UserBasedCF, ItemBasedCF
from src.models.svd_model import SVDRecommender
from src.models.neural_cf import NeuralCFRecommender

def main():
    parser = argparse.ArgumentParser(description="Train Recommendation Models")
    parser.add_argument('--model', type=str, choices=['knn', 'svd', 'neural'], required=True)
    parser.add_argument('--variant', type=str, choices=['user', 'item'], default='user')
    parser.add_argument('--sample-frac', type=float, default=config.SAMPLE_FRACTION)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save-path', type=str, default='data/processed/')
    
    args = parser.parse_args()
    
    parquet_path = os.path.join(config.PROCESSED_DATA_DIR, 'all_ratings.parquet')
    if os.path.exists(parquet_path):
        print(f"Loading data from {parquet_path}")
        df = load_processed(parquet_path)
        sampled_df = sample_data(df, frac=args.sample_frac)
    else:
        print("Processed parquet not found. Parsing raw data...")
        sampled_df = load_all_ratings(sample_frac=args.sample_frac)
        save_processed(sampled_df, parquet_path)
    train_df, test_df = create_train_test_split(sampled_df, strategy='random')
    
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.model in ['knn', 'svd']:
        print("Building Surprise datasets...")
        trainset = build_surprise_dataset(train_df).build_full_trainset()
        
        if args.model == 'svd':
            model = SVDRecommender(n_factors=config.SVD_PARAMS['n_factors'],
                                   n_epochs=config.SVD_PARAMS['n_epochs'],
                                   lr_all=config.SVD_PARAMS['lr_all'],
                                   reg_all=config.SVD_PARAMS['reg_all'])
            model_file = 'svd_model.pkl'
        else:
            if args.variant == 'user':
                model = UserBasedCF(k=config.KNN_USER_PARAMS['k'], sim_name=config.KNN_USER_PARAMS['sim_options']['name'])
                model_file = 'knn_user_model.pkl'
            else:
                model = ItemBasedCF(k=config.KNN_ITEM_PARAMS['k'], sim_name=config.KNN_ITEM_PARAMS['sim_options']['name'])
                model_file = 'knn_item_model.pkl'
                
        model.fit(trainset)
        model.save(os.path.join(args.save_path, model_file))
        print(f"Model saved to {os.path.join(args.save_path, model_file)}")
        
    elif args.model == 'neural':
        print("Encoding categorical IDs for Neural CF...")
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        user_encoder.fit(sampled_df['user_id'])
        item_encoder.fit(sampled_df['movie_id'])
        
        train_ncf = train_df.copy()
        train_ncf['user_idx'] = user_encoder.transform(train_df['user_id'])
        train_ncf['item_idx'] = item_encoder.transform(train_df['movie_id'])
        
        test_ncf = test_df.copy()
        test_ncf['user_idx'] = user_encoder.transform(test_df['user_id'])
        test_ncf['item_idx'] = item_encoder.transform(test_df['movie_id'])
        
        n_users = len(user_encoder.classes_)
        n_items = len(item_encoder.classes_)
        
        model = NeuralCFRecommender(n_users, n_items,
                                    embed_dim=config.NCF_PARAMS['embedding_dim'],
                                    hidden_layers=config.NCF_PARAMS['hidden_layers'],
                                    dropout=config.NCF_PARAMS['dropout'],
                                    lr=config.NCF_PARAMS['lr'],
                                    batch_size=config.NCF_PARAMS['batch_size'])
        model.user_encoder = user_encoder
        model.item_encoder = item_encoder
        
        model.fit(train_ncf, val_df=test_ncf, epochs=args.epochs)
        model.save(os.path.join(args.save_path, 'neural_cf_model.pkl'))
        print(f"Model saved to {os.path.join(args.save_path, 'neural_cf_model.pkl')}")

if __name__ == "__main__":
    main()
