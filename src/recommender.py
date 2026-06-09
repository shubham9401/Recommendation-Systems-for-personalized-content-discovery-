import numpy as np
import pandas as pd

class RecommendationEngine:
    def __init__(self, model, movie_titles_df, user_encoder=None, item_encoder=None):
        self.model = model
        if 'movie_id' in movie_titles_df.columns:
            self.titles_df = movie_titles_df.set_index('movie_id')
        else:
            self.titles_df = movie_titles_df
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def generate_top_k(self, user_id, k=10, exclude_rated=True, rated_items=None):
        all_items = self.titles_df.index.tolist()
        
        if hasattr(self.model, 'get_top_n'):
            kwargs = {'user_rated_items': rated_items} if 'user_rated_items' in self.model.get_top_n.__code__.co_varnames else {}
            top_preds = self.model.get_top_n(user_id, all_items, n=k, exclude_rated=exclude_rated, **kwargs)
        else:
            top_preds = []
            for iid in all_items:
                if exclude_rated and rated_items and iid in rated_items:
                    continue
                est = self.model.predict(user_id, iid)
                top_preds.append((iid, est))
            top_preds.sort(key=lambda x: x[1], reverse=True)
            top_preds = top_preds[:k]
            
        results = []
        for iid, est in top_preds:
            title = self.titles_df.loc[iid, 'title'] if iid in self.titles_df.index else "Unknown"
            year = self.titles_df.loc[iid, 'year'] if iid in self.titles_df.index else np.nan
            results.append((iid, est, title, year))
            
        return results

    def display_recommendations(self, user_id, recs):
        print(f"Top Recommendations for User {user_id}:")
        print("-" * 65)
        for rank, (iid, est, title, year) in enumerate(recs, 1):
            stars = "★" * int(round(est)) + "☆" * (5 - int(round(est)))
            year_str = f"({int(year)})" if not pd.isna(year) else ""
            print(f"{rank:2d}. {str(title)[:40]:<40} {year_str:<6} | {stars} ({est:.2f})")
        print("-" * 65)

    def explain_recommendation(self, user_id, movie_id):
        model_type = self.model.__class__.__name__
        title = self.titles_df.loc[movie_id, 'title'] if movie_id in self.titles_df.index else str(movie_id)
        
        if 'KNN' in model_type or 'UserBasedCF' in model_type or 'ItemBasedCF' in model_type:
            return f"Because you liked [neighbor movies], similar users also enjoyed '{title}'"
        elif 'SVD' in model_type:
            return f"This matches your taste profile — similar latent factors to [top user movies]"
        elif 'Neural' in model_type or 'NeuMF' in model_type:
            return f"Based on your rating history, '{title}' fits your preferences"
        else:
            return f"Recommended based on overall system popularity and similarity."

    def get_similar_items(self, movie_id, k=10):
        if hasattr(self.model, 'get_similar_items'):
            sims = self.model.get_similar_items(movie_id, k=k)
            results = []
            for iid, sim in sims:
                title = self.titles_df.loc[iid, 'title'] if iid in self.titles_df.index else "Unknown"
                results.append((iid, sim, title))
            return results
            
        elif hasattr(self.model, 'get_item_factors'):
            target_vec = self.model.get_item_factors(movie_id)
            if target_vec is None:
                return []
                
            sims = []
            for iid in self.titles_df.index:
                if iid == movie_id: continue
                vec = self.model.get_item_factors(iid)
                if vec is not None:
                    norm = np.linalg.norm(target_vec) * np.linalg.norm(vec)
                    cos_sim = np.dot(target_vec, vec) / norm if norm > 0 else 0
                    sims.append((iid, cos_sim))
                    
            sims.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for iid, sim in sims[:k]:
                title = self.titles_df.loc[iid, 'title'] if iid in self.titles_df.index else "Unknown"
                results.append((iid, sim, title))
            return results
        else:
            print("Model does not support item similarity extraction.")
            return []

    def get_user_history(self, user_id, df, n=10):
        user_history = df[df['user_id'] == user_id].sort_values('rating', ascending=False).head(n)
        titles_reset = self.titles_df.reset_index()
        user_history = user_history.merge(titles_reset, on='movie_id', how='left')
        return user_history[['movie_id', 'title', 'year', 'rating', 'date']]
