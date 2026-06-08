import os
import time
import pickle
from surprise import KNNWithMeans, PredictionImpossible

class UserBasedCF:
    def __init__(self, k=40, sim_name='cosine', min_k=5):
        self.algo = KNNWithMeans(k=k, min_k=min_k, sim_options={'name': sim_name, 'user_based': True, 'min_support': 3})
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        start_time = time.time()
        print("Training UserBasedCF...")
        self.algo.fit(trainset)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def predict(self, user_id, item_id):
        try:
            pred = self.algo.predict(user_id, item_id)
            if pred.details.get('was_impossible', False):
                return self.trainset.global_mean
            return pred.est
        except Exception:
            return self.trainset.global_mean if self.trainset else 3.0

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

    def get_similar_users(self, user_id, k=10):
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            neighbors_inner = self.algo.get_neighbors(inner_uid, k=k)
            similarities = [(self.trainset.to_raw_uid(inner_id), self.algo.sim[inner_uid, inner_id]) 
                            for inner_id in neighbors_inner]
            return similarities
        except ValueError:
            return []

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)


class ItemBasedCF:
    def __init__(self, k=40, sim_name='pearson', min_k=5):
        self.algo = KNNWithMeans(k=k, min_k=min_k, sim_options={'name': sim_name, 'user_based': False, 'min_support': 3})
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        start_time = time.time()
        print("Training ItemBasedCF...")
        self.algo.fit(trainset)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def predict(self, user_id, item_id):
        try:
            pred = self.algo.predict(user_id, item_id)
            if pred.details.get('was_impossible', False):
                return self.trainset.global_mean
            return pred.est
        except Exception:
            return self.trainset.global_mean if self.trainset else 3.0

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

    def get_similar_items(self, item_id, k=10):
        try:
            inner_iid = self.trainset.to_inner_iid(item_id)
            neighbors_inner = self.algo.get_neighbors(inner_iid, k=k)
            similarities = [(self.trainset.to_raw_iid(inner_id), self.algo.sim[inner_iid, inner_id]) 
                            for inner_id in neighbors_inner]
            return similarities
        except ValueError:
            return []

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
