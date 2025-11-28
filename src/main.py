import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import numpy as np
import os 
import sys 
import time
import random 

try:
    SCRIPT_PATH = os.path.abspath(__file__)
    SRC_DIR = os.path.dirname(SCRIPT_PATH)
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
except NameError:
    DATA_DIR = "../data"

print(f"Data directory is set to: {DATA_DIR}")

def load_ratings_sample(sample_size=300_000):
    """
    Reads a *random* 300k chunk of ratings.json.
    """
    print(f"Loading and splitting rating data (RANDOM {sample_size} chunk)...")
    ratings_file = os.path.join(DATA_DIR, "ratings.json")
    
    TOTAL_LINES = 13_005_494
    total_chunks = TOTAL_LINES // sample_size
    random_chunk_index = random.randint(0, total_chunks - 1)
    print(f"Total of {total_chunks} chunks available. Selecting random chunk #{random_chunk_index}...")

    try:
        chunk_iter = pd.read_json(
            ratings_file, 
            lines=True, 
            chunksize=sample_size,
            dtype={'user_id': 'int32', 'item_id': 'int32', 'rating': 'float32'}
        )
        
        ratings_df = None
        for i, chunk in enumerate(chunk_iter):
            if i == random_chunk_index:
                ratings_df = chunk
                print(f"Successfully read {len(ratings_df)} ratings from chunk {i}.")
                break
        
        if ratings_df is None: raise Exception("Could not read random chunk.")
        
    except Exception as e:
        print(f"Error reading ratings.json: {e}")
        return None
        
    ratings_df = ratings_df[['user_id', 'item_id', 'rating']]
    ratings_df = ratings_df.rename(columns={'user_id': 'userId', 'item_id': 'movieId'})
    return ratings_df

def build_content_profiles(relevant_movie_ids):
    """
    Builds content profiles ONLY for movies in the training list.
    """
    print("Building content profiles...")
    tags_file = os.path.join(DATA_DIR, "tags.json")
    tag_apps_file = os.path.join(DATA_DIR, "tag_count.json")
    
    try:
        tags_df = pd.read_json(tags_file, lines=True) 
        
        chunk_iter = pd.read_json(
            tag_apps_file, 
            lines=True, 
            chunksize=1_000_000,
            dtype={'item_id': 'int32', 'tag_id': 'int32'}
        )
        
        tag_apps_df_list = []
        print("Filtering tag applications based on TRAINING movie IDs...")
        for chunk in chunk_iter:
            relevant_chunk = chunk[chunk['item_id'].isin(relevant_movie_ids)]
            tag_apps_df_list.append(relevant_chunk)
        
        tag_apps_df = pd.concat(tag_apps_df_list, ignore_index=True)
        print(f"Found {len(tag_apps_df)} tag applications for relevant movies.")

        print("Merging tag and tag application data...")
        merged_tags = pd.merge(
            tag_apps_df, 
            tags_df, 
            left_on='tag_id',
            right_on='id',
            how='left'
        )

        print("Grouping tags into movie profiles...")
        movie_profiles = merged_tags.dropna(subset=['tag']).groupby('item_id')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        
        movie_profiles = movie_profiles.rename(columns={'item_id': 'movieId', 'tag': 'profile_text'})
        movie_profiles = movie_profiles.set_index('movieId')
    
    except Exception as e:
        print(f"Tag-based profile building failed: {e}")
        return None

    print(f"{len(movie_profiles)} movie profiles built successfully using tags.")
    return movie_profiles

def build_similarity_matrix(movie_profiles):
    print("Calculating TF-IDF and Similarity Matrix...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000) 
    tfidf_matrix = tfidf.fit_transform(movie_profiles['profile_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_df = pd.DataFrame(cosine_sim, index=movie_profiles.index, columns=movie_profiles.index)
    print("Similarity matrix completed.")
    return sim_df

def predict_rating(user_id, movie_id, train_user_ratings, sim_df, k=30):

    if movie_id not in sim_df:
        return np.nan 
        
    movie_sims = sim_df[movie_id]
    user_rated_items = train_user_ratings.get(user_id, [])
    weighted_sum = 0
    total_similarity = 0
    scores = []
    for rated_movie_id, rating in user_rated_items:
        if rated_movie_id in movie_sims:
            scores.append((movie_sims[rated_movie_id], rating))
    scores = sorted([s for s in scores if s[0] > 0], key=lambda x: x[0], reverse=True)
    for sim, rating in scores[:k]:
        weighted_sum += sim * rating
        total_similarity += sim
    if total_similarity == 0:
        return np.nan
    return weighted_sum / total_similarity

def evaluate_mae(test_df, train_user_ratings, sim_df):
    print("\nCalculating MAE...")
    predictions = []
    true_ratings = []
    
    for row in test_df.itertuples(index=False):
        pred = predict_rating(row.userId, row.movieId, train_user_ratings, sim_df)
        if not np.isnan(pred):
            predictions.append(pred)
            true_ratings.append(row.rating)
    if not predictions:
        print("Could not make any predictions.")
        return
    mae = mean_absolute_error(true_ratings, predictions)
    print(f"--- MAE Result ---")
    print(f"Total predictions made: {len(predictions)} (out of {len(test_df)} test samples)")
    print(f"Calculated MAE: {mae:.4f}")

def calculate_hit_ratio_vectorized(train_df, test_df, sim_df, N=10):
    """
    Calculates Hit Ratio using the fast vectorized .dot() product method
    shown in the professor's content_based.ipynb notebook .
    """
    print(f"\nCalculating Hit Ratio (N={N}) using Vectorized Method...")
    
    test_user_hits = test_df[test_df['rating'] >= 4.0].groupby('userId')['movieId'].apply(list)
    if test_user_hits.empty:
        print("No users with high ratings found in the test set.")
        return

    known_movie_ids = set(sim_df.index)
    
    train_df_filtered = train_df[train_df['movieId'].isin(known_movie_ids)]
    
    train_user_ratings_matrix = pd.pivot_table(
        train_df_filtered, 
        values='rating', 
        index='movieId', 
        columns='userId'
    ).fillna(0)
    
    print("Calculating recommendation scores for all users via matrix multiplication...")

    recommendation_scores_matrix = sim_df.dot(train_user_ratings_matrix)

    hits = 0
    total_users = 0

    for user_id, true_hits in test_user_hits.items():
        if user_id not in recommendation_scores_matrix.columns:
            continue
            
        total_users += 1
        
        user_scores = recommendation_scores_matrix[user_id]
        
        seen_movies = train_df_filtered[train_df_filtered['userId'] == user_id]['movieId'].unique()
        user_scores = user_scores.drop(seen_movies, errors='ignore')
        
        top_n_recs = user_scores.nlargest(N).index
        
        if any(movie_id in top_n_recs for movie_id in true_hits):
            hits += 1

    hit_ratio = hits / total_users if total_users > 0 else 0
    print(f"--- Hit Ratio Result ---")
    print(f"Analyzed {total_users} test users")
    print(f"Total Hits: {hits}")
    print(f"Hit Ratio @{N}: {hit_ratio:.4f}")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory can not be found. Looked for: {DATA_DIR}")
        return

    start_time = time.time()

    ratings_df = load_ratings_sample(sample_size=300_000)
    if ratings_df is None:
        print("Failed loading ratings data.")
        return

    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} rows, Test set: {len(test_df)} rows.")

    train_user_ratings = defaultdict(list)
    for _, row in train_df.iterrows():
        train_user_ratings[row['userId']].append((row['movieId'], row['rating']))
        
    relevant_ids = set(train_df['movieId'].unique())
    print(f"Found {len(relevant_ids)} unique movies in the TRAINING set.")
    
    main_movie_profiles = build_content_profiles(relevant_ids)
    if main_movie_profiles is None:
        print("Failed building movie profiles.")
        return

    main_sim_df = build_similarity_matrix(main_movie_profiles)
    
    evaluate_mae(test_df, train_user_ratings, main_sim_df) 
    
    calculate_hit_ratio_vectorized(train_df, test_df, main_sim_df, N=10)
    
    print("\nExecution finished.")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total runtime is {total_runtime:.2f} seconds.")

if __name__ == "__main__":
    main()
