import os
import sys
import argparse
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_processed
from src.preprocessing import sample_data, create_train_test_split, build_surprise_dataset
from src.evaluation import evaluate_all
from src.models.knn_cf import UserBasedCF, ItemBasedCF
from src.models.svd_model import SVDRecommender
from src.models.neural_cf import NeuralCFRecommender

def main():
    parser = argparse.ArgumentParser(description="Evaluate Recommendation Models")
    parser.add_argument('--model', type=str, choices=['knn', 'svd', 'neural'], required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--metrics', nargs='+', choices=['rmse','mae','map10','precision','recall','ndcg','coverage'], 
                        default=['rmse','map10'])
    parser.add_argument('--k', type=int, default=10)
    
    args = parser.parse_args()
    
    parquet_path = os.path.join(config.PROCESSED_DATA_DIR, 'all_ratings.parquet')
    if not os.path.exists(parquet_path):
        print("Processed data not found. Run train.py first.")
        return
        
    df = load_processed(parquet_path)
    sampled_df = sample_data(df, frac=config.SAMPLE_FRACTION)
    train_df, test_df = create_train_test_split(sampled_df, strategy='temporal')
    
    print(f"Loading {args.model} model from {args.model_path}...")
    
    if args.model == 'svd':
        model = SVDRecommender()
        model.load(args.model_path)
    elif args.model == 'knn':
        model = UserBasedCF() 
        model.load(args.model_path)
    elif args.model == 'neural':
        with open(args.model_path, 'rb') as f:
            data = pickle.load(f)
            user_encoder = data['user_encoder']
            item_encoder = data['item_encoder']
            n_users = len(user_encoder.classes_)
            n_items = len(item_encoder.classes_)
        model = NeuralCFRecommender(n_users, n_items, embed_dim=config.NCF_PARAMS['embedding_dim'])
        model.load(args.model_path)
        
    if args.model in ['knn', 'svd']:
        trainset = build_surprise_dataset(train_df).build_full_trainset()
        testset_tuples = build_surprise_dataset(test_df).build_full_trainset().build_testset()
        evaluate_all(model, testset_tuples, trainset, k=args.k)
    else:
        testset_tuples = list(test_df[['user_id', 'movie_id', 'rating']].itertuples(index=False, name=None))
        evaluate_all(model, testset_tuples, train_df, k=args.k)

if __name__ == "__main__":
    main()
