RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
RANDOM_SEED = 42
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 3.5
TOP_K = 10
SAMPLE_FRACTION = 0.1

# KNN hyperparams: K=40, sim='cosine' for user-based, 'pearson' for item-based
KNN_USER_PARAMS = {
    'k': 40,
    'sim_options': {'name': 'cosine', 'user_based': True}
}

KNN_ITEM_PARAMS = {
    'k': 40,
    'sim_options': {'name': 'pearson', 'user_based': False}
}

# SVD hyperparams: n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02
SVD_PARAMS = {
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02
}

# Neural CF hyperparams: embedding_dim=64, hidden_layers=[128,64,32], dropout=0.2, lr=0.001, batch_size=1024
NCF_PARAMS = {
    'embedding_dim': 64,
    'hidden_layers': [128, 64, 32],
    'dropout': 0.2,
    'lr': 0.001,
    'batch_size': 1024
}
