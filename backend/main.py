from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from rapidfuzz import process, fuzz
import os
import math

app = FastAPI(title="CineMatch AI API", description="Backend for the movie recommendation system.")

# Configure CORS so the frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_file_path = os.path.join(BASE_DIR, 'data', 'movies.csv')
ratings_file_path = os.path.join(BASE_DIR, 'data', 'ratings.csv')
links_file_path = os.path.join(BASE_DIR, 'data', 'links.csv')

movies_data = pd.DataFrame()
hybrid_sim = []

def load_data():
    global movies_data, hybrid_sim
    try:
        movies_df = pd.read_csv(movies_file_path)
        ratings_df = pd.read_csv(ratings_file_path)
        links_df = pd.read_csv(links_file_path)
        
        # Merge links to fetch tmdbId for frontend image fetching
        movies_data = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
        
        # 1. Content-Based Similarity (TF-IDF on Genres)
        movies_data['genres_processed'] = movies_data['genres'].str.replace('|', ' ', regex=False)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['genres_processed'])
        content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 2. Collaborative Filtering Similarity (TruncatedSVD on User-Item Matrix)
        # This solves the Cold Start effectively using an Item-Item collaborative approach
        user_item = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        # Ensure our CF matrix columns perfectly align with the movies_data index
        user_item = user_item.reindex(columns=movies_data['movieId'].values, fill_value=0)
        
        svd = TruncatedSVD(n_components=20, random_state=42)
        item_embeddings = svd.fit_transform(user_item.T)
        cf_sim = cosine_similarity(item_embeddings)
        
        # 3. Hybrid Recommendation Engine
        # We average the content similarity and collaborative similarity for a robust score
        hybrid_sim = (content_sim + cf_sim) / 2.0
        
        print("Data loaded and Hybrid models initialized successfully. (MovieLens Dataset)")
    except Exception as e:
        print(f"Error loading datasets: {e}")

# Load data on startup
load_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to CineMatch AI API"}

@app.get("/api/movies/search")
def search_movies(query: str):
    if movies_data.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    # Semantic/Fuzzy search using rapidfuzz
    matches = process.extract(query, movies_data['title'], scorer=fuzz.WRatio, limit=5)
    results = []
    for match in matches:
        title = match[0]
        score = match[1]
        movie = movies_data[movies_data['title'] == title].iloc[0]
        
        tmdb_id = None
        if not pd.isna(movie['tmdbId']):
            tmdb_id = int(movie['tmdbId'])
            
        results.append({
            "id": int(movie['movieId']),
            "tmdbId": tmdb_id,
            "title": movie['title'],
            "genres": movie['genres'],
            "match_score": score
        })
    return {"results": results}

@app.get("/api/recommendations")
def get_recommendations(movie_name: str, n_recs: int = 5):
    if movies_data.empty:
        raise HTTPException(status_code=500, detail="Data not available")

    # Find closest movie in dataset
    closest_match = process.extractOne(movie_name, movies_data['title'], scorer=fuzz.WRatio)
    
    if not closest_match or closest_match[1] < 50:
        raise HTTPException(status_code=404, detail="Could not find a similar movie.")
    
    movie_title = closest_match[0]
    movie_index = movies_data[movies_data['title'] == movie_title].index[0]

    # Calculate hybrid similarity
    sim_scores = list(enumerate(hybrid_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the movie itself (index 0) and get top n_recs
    top_movie_indices = [i[0] for i in sim_scores[1:n_recs + 1]]
    
    recommended_movies = []
    for idx in top_movie_indices:
        movie = movies_data.iloc[idx]
        
        tmdb_id = None
        if not pd.isna(movie['tmdbId']):
            tmdb_id = int(movie['tmdbId'])
            
        recommended_movies.append({
            "id": int(movie['movieId']),
            "tmdbId": tmdb_id,
            "title": movie['title'],
            "genres": movie['genres']
        })
        
    return {
        "target_movie": movie_title,
        "recommendations": recommended_movies
    }
