import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_processed, load_movie_titles
from src.recommender import RecommendationEngine
from src.models.svd_model import SVDRecommender
from src.models.knn_cf import UserBasedCF, ItemBasedCF
from src.models.neural_cf import NeuralCFRecommender

st.set_page_config(page_title="Netflix Recs", layout="wide")

@st.cache_data
def load_data():
    parquet_path = os.path.join(config.PROCESSED_DATA_DIR, 'all_ratings.parquet')
    titles_path = os.path.join(config.RAW_DATA_DIR, 'movie_titles.txt')
    
    if not os.path.exists(parquet_path):
        st.warning(f"Data not found at {parquet_path}. Please run scripts/train.py first.")
        st.stop()
        
    df = load_processed(parquet_path)
    titles = load_movie_titles(titles_path)
    return df, titles

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(config.PROCESSED_DATA_DIR, model_name)
    if not os.path.exists(model_path):
        st.warning(f"Model file '{model_name}' not found. Please run scripts/train.py first.")
        st.stop()
        
    if 'svd' in model_name:
        model = SVDRecommender()
        model.load(model_path)
    elif 'knn_user' in model_name:
        model = UserBasedCF()
        model.load(model_path)
    elif 'knn_item' in model_name:
        model = ItemBasedCF()
        model.load(model_path)
    elif 'neural' in model_name:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            u_enc = data['user_encoder']
            i_enc = data['item_encoder']
        model = NeuralCFRecommender(len(u_enc.classes_), len(i_enc.classes_), embed_dim=config.NCF_PARAMS['embedding_dim'])
        model.load(model_path)
    else:
        st.error("Unknown model type")
        st.stop()
        
    return model

def main():
    st.sidebar.title("Netflix Recommendation Engine")
    
    model_choice = st.sidebar.radio("Select Model", 
                                    ["SVD", "KNN (User)", "KNN (Item)", "Neural CF"])
    
    page = st.sidebar.radio("Navigation", 
                            ["Get Recommendations", "Explore Movies", "Model Comparison"])
    
    df, titles = load_data()
    
    # Map model choice to file
    model_file_map = {
        "SVD": "svd_model.pkl",
        "KNN (User)": "knn_user_model.pkl",
        "KNN (Item)": "knn_item_model.pkl",
        "Neural CF": "neural_cf_model.pkl"
    }
    
    model = load_model(model_file_map[model_choice])
    rec_engine = RecommendationEngine(model, titles)
    
    if page == "Get Recommendations":
        st.header("Get Personalized Recommendations")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Let user select a user_id from the top users for demo purposes
            top_users = df['user_id'].value_counts().head(200).index.tolist()
            user_id = st.selectbox("Select User ID (Top 200 active users)", top_users)
            k = st.slider("Number of recommendations (Top-K)", 5, 20, 10)
            
            if st.button("Generate Recommendations"):
                with st.spinner("Generating..."):
                    history = rec_engine.get_user_history(user_id, df, n=10)
                    
                    rated_items = set(df[df['user_id'] == user_id]['movie_id'])
                    recs = rec_engine.generate_top_k(user_id, k=k, exclude_rated=True, rated_items=rated_items)
                    
                    st.session_state['history'] = history
                    st.session_state['recs'] = recs
                    st.session_state['user_id'] = user_id
                    
        with col2:
            if 'recs' in st.session_state and st.session_state['user_id'] == user_id:
                st.subheader("User's Top Past Ratings")
                st.dataframe(st.session_state['history'][['title', 'year', 'rating']], hide_index=True)
                
                st.subheader(f"Top {k} Recommendations")
                
                rec_df_data = []
                for rank, (iid, est, title, year) in enumerate(st.session_state['recs'], 1):
                    stars = "★" * int(round(est)) + "☆" * (5 - int(round(est)))
                    rec_df_data.append({
                        "Rank": rank,
                        "Movie Title": title,
                        "Year": int(year) if not pd.isna(year) else "N/A",
                        "Predicted Rating": f"{est:.2f} {stars}",
                        "Score": est,
                        "Explanation": rec_engine.explain_recommendation(user_id, iid)
                    })
                
                rec_df = pd.DataFrame(rec_df_data)
                st.table(rec_df[['Rank', 'Movie Title', 'Year', 'Predicted Rating']])
                
                st.subheader("Predicted Scores")
                chart_df = rec_df.set_index('Movie Title')[['Score']].sort_values('Score')
                # Use bar_chart wrapper provided by streamlit
                st.bar_chart(chart_df, horizontal=True)
                
                st.subheader("Why these?")
                for item in rec_df_data:
                    st.write(f"**{item['Movie Title']}**: {item['Explanation']}")
                    
    elif page == "Explore Movies":
        st.header("Explore Movies")
        
        movie_title = st.selectbox("Select a Movie", titles['title'].dropna().unique())
        movie_id = titles[titles['title'] == movie_title]['movie_id'].values[0]
        
        movie_stats = df[df['movie_id'] == movie_id]['rating'].agg(['mean', 'count'])
        year = titles[titles['movie_id'] == movie_id]['year'].values[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Release Year", int(year) if not pd.isna(year) else "Unknown")
        col2.metric("Average Rating", f"{movie_stats['mean']:.2f}")
        col3.metric("Number of Ratings", int(movie_stats['count']))
        
        st.subheader("Rating Distribution")
        movie_ratings = df[df['movie_id'] == movie_id]['rating']
        dist = movie_ratings.value_counts().sort_index()
        st.bar_chart(dist)
        
        st.subheader("Similar Movies")
        # We can extract similar items if the model supports it
        if model_choice in ["SVD", "KNN (Item)", "Neural CF"]:
            sims = rec_engine.get_similar_items(movie_id, k=10)
            if sims:
                sim_df = pd.DataFrame(sims, columns=["ID", "Similarity Score", "Title"])
                st.table(sim_df[['Title', 'Similarity Score']])
            else:
                st.write("Model does not support item similarity extraction or data unavailable.")
                
    elif page == "Model Comparison":
        st.header("Model Comparison")
        st.markdown("""
        ### Evaluation Metrics
        Comparing the models across key metrics. (Using pre-computed baseline values if `evaluation_results.csv` is absent).
        """)
        
        metrics_file = os.path.join(config.PROCESSED_DATA_DIR, 'evaluation_results.csv')
        if os.path.exists(metrics_file):
            results_df = pd.read_csv(metrics_file)
            st.dataframe(results_df, hide_index=True)
            
            st.subheader("RMSE Comparison")
            chart_data = results_df.set_index('Model')[['RMSE']]
            st.bar_chart(chart_data)
        else:
            st.warning("Evaluation metrics not found. Please run scripts/evaluate.py to generate evaluation_results.csv.")
        
        st.markdown("""
        ### Trade-offs
        * **SVD**: Best overall accuracy (lowest RMSE), robust coverage, and fastest inference time. Strongest baseline for production.
        * **KNN**: Good for interpretability ("Because you watched X"), but suffers from extreme sparsity and slow inference.
        * **Neural CF**: Can capture complex non-linear patterns but requires extensive hyperparameter tuning and is prone to overfitting on sparse datasets.
        """)

if __name__ == "__main__":
    main()
