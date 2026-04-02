from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import os

app = FastAPI(title="CineMatch AI API", description="Backend for the movie recommendation system.")

# Configure CORS so the frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since this is a local project/prototype
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_file_path = os.path.join(BASE_DIR, 'data', 'movies.csv')
ratings_file_path = os.path.join(BASE_DIR, 'data', 'ratings.csv')

movies_data = pd.DataFrame()
cosine_sim = []

def load_data():
    global movies_data, cosine_sim
    try:
        movies_data = pd.read_csv(movies_file_path)
        
        # Precompute TF-IDF matrix & Cosine Similarity
        movies_data['genres_processed'] = movies_data['genres'].str.replace('|', ' ', regex=False)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['genres_processed'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Data loaded and models initialized successfully.")
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
    # We find top 5 matches
    matches = process.extract(query, movies_data['title'], scorer=fuzz.WRatio, limit=5)
    results = []
    for match in matches:
        title = match[0]
        score = match[1]
        movie = movies_data[movies_data['title'] == title].iloc[0]
        results.append({
            "id": int(movie['movieId']),
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
        raise HTTPException(status_code=404, detail="Could not find a similar movie to base recommendations on.")
    
    movie_title = closest_match[0]
    movie_index = movies_data[movies_data['title'] == movie_title].index[0]

    # Calculate similarity
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the movie itself (index 0) and get top n_recs
    top_movie_indices = [i[0] for i in sim_scores[1:n_recs + 1]]
    
    recommended_movies = []
    for idx in top_movie_indices:
        movie = movies_data.iloc[idx]
        recommended_movies.append({
            "id": int(movie['movieId']),
            "title": movie['title'],
            "genres": movie['genres']
        })
        
    return {
        "target_movie": movie_title,
        "recommendations": recommended_movies
    }
