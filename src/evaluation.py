import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, ndcg_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def rmse(predictions):
    if not predictions:
        return 0.0
    true_ratings = [p.r_ui for p in predictions]
    est_ratings = [p.est for p in predictions]
    return root_mean_squared_error(true_ratings, est_ratings)

def mae(predictions):
    if not predictions:
        return 0.0
    true_ratings = [p.r_ui for p in predictions]
    est_ratings = [p.est for p in predictions]
    return mean_absolute_error(true_ratings, est_ratings)

def precision_at_k(user_predictions_dict, k=10, threshold=config.RELEVANCE_THRESHOLD):
    precisions = []
    for uid, user_preds in user_predictions_dict.items():
        user_preds.sort(key=lambda x: x[1], reverse=True)
        top_k = user_preds[:k]
        n_rel = sum((true_r >= threshold) for (_, _, true_r) in top_k if true_r is not None)
        precisions.append(n_rel / k)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(user_predictions_dict, k=10, threshold=config.RELEVANCE_THRESHOLD):
    recalls = []
    for uid, user_preds in user_predictions_dict.items():
        user_preds.sort(key=lambda x: x[1], reverse=True)
        top_k = user_preds[:k]
        
        n_rel_in_top_k = sum((true_r >= threshold) for (_, _, true_r) in top_k if true_r is not None)
        n_rel_total = sum((true_r >= threshold) for (_, _, true_r) in user_preds if true_r is not None)
        
        if n_rel_total > 0:
            recalls.append(n_rel_in_top_k / n_rel_total)
    return np.mean(recalls) if recalls else 0.0

def map_at_k(user_predictions_dict, k=10, threshold=config.RELEVANCE_THRESHOLD):
    aps = []
    evaluated_users = 0
    for uid, user_preds in user_predictions_dict.items():
        n_rel_total = sum((true_r >= threshold) for (_, _, true_r) in user_preds if true_r is not None)
        if n_rel_total == 0:
            continue
            
        evaluated_users += 1
        user_preds.sort(key=lambda x: x[1], reverse=True)
        top_k = user_preds[:k]
        
        ap = 0.0
        hits = 0
        for i, (_, _, true_r) in enumerate(top_k):
            if true_r is not None and true_r >= threshold:
                hits += 1
                ap += hits / (i + 1.0)
        
        aps.append(ap / min(n_rel_total, k))
        
    print(f"Evaluated MAP@{k} for {evaluated_users} users with relevant items.")
    return np.mean(aps) if aps else 0.0

def ndcg_at_k(user_predictions_dict, k=10):
    ndcgs = []
    for uid, user_preds in user_predictions_dict.items():
        eval_items = [(est_r, true_r) for (_, est_r, true_r) in user_preds if true_r is not None]
        if len(eval_items) < 2:
            continue
            
        eval_items.sort(key=lambda x: x[0], reverse=True)
        est_r = [x[0] for x in eval_items[:k]]
        true_r = [x[1] for x in eval_items[:k]]
        
        try:
            score = ndcg_score([true_r], [est_r], k=k)
            ndcgs.append(score)
        except ValueError:
            pass
            
    return np.mean(ndcgs) if ndcgs else 0.0

def coverage(recommended_items_set, n_total_items):
    if n_total_items == 0:
        return 0.0
    return len(recommended_items_set) / n_total_items

def build_user_predictions_dict(model, testset, trainset=None):
    print("Building predictions dictionary across all unrated items for test users...")
    user_predictions_dict = defaultdict(list)
    
    test_user_true_ratings = defaultdict(dict)
    if hasattr(testset, 'ur'):
        for u, i, r in testset:
            test_user_true_ratings[u][i] = r
    else:
        for tup in testset:
            if hasattr(tup, 'uid'):
                test_user_true_ratings[tup.uid][tup.iid] = tup.r_ui
            else:
                test_user_true_ratings[tup[0]][tup[1]] = tup[2]
                
    test_users = list(test_user_true_ratings.keys())
    
    all_items = set()
    user_rated_train = defaultdict(set)
    
    if trainset:
        if hasattr(trainset, 'all_items'): 
            for iid in trainset.all_items():
                all_items.add(trainset.to_raw_iid(iid))
            for uid, iid_rating in trainset.ur.items():
                raw_uid = trainset.to_raw_uid(uid)
                for inner_iid, _ in iid_rating:
                    user_rated_train[raw_uid].add(trainset.to_raw_iid(inner_iid))
        elif isinstance(trainset, pd.DataFrame): 
            all_items = set(trainset['movie_id'].unique())
            for _, row in trainset.iterrows():
                user_rated_train[row['user_id']].add(row['movie_id'])

    if not all_items:
        for tup in testset:
            iid = tup.iid if hasattr(tup, 'iid') else tup[1]
            all_items.add(iid)
            
    for u in test_users:
        rated_in_train = user_rated_train.get(u, set())
        unrated_items = all_items - rated_in_train
        
        if hasattr(model, 'get_top_n'):
            top_k = model.get_top_n(u, list(unrated_items), n=len(unrated_items), exclude_rated=True)
            for iid, est in top_k:
                true_r = test_user_true_ratings[u].get(iid, None)
                user_predictions_dict[u].append((iid, est, true_r))
        else:
            for iid in unrated_items:
                if hasattr(model, 'predict') and 'details' in str(type(model.predict)):
                    est = model.predict(u, iid).est 
                else:
                    est = model.predict(u, iid) 
                true_r = test_user_true_ratings[u].get(iid, None)
                user_predictions_dict[u].append((iid, est, true_r))
                
    return user_predictions_dict

def evaluate_all(model, testset, trainset=None, k=10):
    print("Running evaluation suite...")
    
    if isinstance(testset, list) and hasattr(testset[0], 'est'):
        preds = testset
    else:
        preds = []
        for tup in testset:
            if hasattr(tup, 'uid'):
                u, i, r = tup.uid, tup.iid, tup.r_ui
            else:
                u, i, r = tup[0], tup[1], tup[2]
                
            if hasattr(model, 'predict') and 'details' in str(type(model.predict)):
                est = model.predict(u, i).est
            else:
                est = model.predict(u, i)
                
            class Pred:
                def __init__(self, u, i, r, est):
                    self.uid = u
                    self.iid = i
                    self.r_ui = r
                    self.est = est
            preds.append(Pred(u, i, r, est))
            
    val_rmse = rmse(preds)
    val_mae = mae(preds)
    
    user_pred_dict = build_user_predictions_dict(model, testset, trainset)
    
    val_prec = precision_at_k(user_pred_dict, k)
    val_rec = recall_at_k(user_pred_dict, k)
    val_map = map_at_k(user_pred_dict, k)
    val_ndcg = ndcg_at_k(user_pred_dict, k)
    
    recommended = set()
    for u, preds_list in user_pred_dict.items():
        preds_list.sort(key=lambda x: x[1], reverse=True)
        for iid, _, _ in preds_list[:k]:
            recommended.add(iid)
            
    total_items = 0
    if trainset and hasattr(trainset, 'all_items'):
        total_items = trainset.n_items
    elif isinstance(trainset, pd.DataFrame):
        total_items = trainset['movie_id'].nunique()
        
    val_cov = coverage(recommended, total_items)
    
    results = {
        'rmse': val_rmse,
        'mae': val_mae,
        f'precision@{k}': val_prec,
        f'recall@{k}': val_rec,
        f'map@{k}': val_map,
        f'ndcg@{k}': val_ndcg,
        'coverage': val_cov
    }
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    for metric, value in results.items():
        print(f"{metric.upper():<15}: {value:.4f}")
    print("="*30 + "\n")
    
    return results
