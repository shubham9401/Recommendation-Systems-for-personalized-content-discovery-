import time
import pickle
import numpy as np
from surprise import SVD
from surprise.model_selection import GridSearchCV

class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, 
                        lr_all=lr_all, reg_all=reg_all, random_state=42, verbose=False)
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        start_time = time.time()
        print(f"Training SVDRecommender (n_factors={self.algo.n_factors})...")
        self.algo.fit(trainset)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def predict(self, user_id, item_id):
        pred = self.algo.predict(user_id, item_id)
        if pred.details.get('was_impossible', False):
            return self.trainset.global_mean
        return pred.est

    def get_top_n(self, user_id, all_item_ids, n=10, exclude_rated=True):
        if self.trainset is None:
            raise ValueError("Model must be trained before predicting.")
            
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
            rated_items = set(j for (j, _) in self.trainset.ur[inner_user_id])
        except ValueError:
            rated_items = set()

        predictions = []
        for item_id in all_item_ids:
            try:
                inner_item_id = self.trainset.to_inner_iid(item_id)
                if exclude_rated and inner_item_id in rated_items:
                    continue
            except ValueError:
                pass
                
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def get_user_factors(self, user_id):
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            return self.algo.pu[inner_uid]
        except ValueError:
            return None

    def get_item_factors(self, item_id):
        try:
            inner_iid = self.trainset.to_inner_iid(item_id)
            return self.algo.qi[inner_iid]
        except ValueError:
            return None

    def tune(self, data, param_grid=None):
        if param_grid is None:
            param_grid = {'n_factors': [50, 100, 150], 'reg_all': [0.02, 0.05, 0.1]}
            
        print("Tuning SVD hyperparameters via GridSearchCV...")
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        
        print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
        print(f"Best Params: {gs.best_params['rmse']}")
        
        self.algo = gs.best_estimator['rmse']
        return gs.best_params['rmse']

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
