import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

ratings_file_path = 'ratings.csv'  
movies_file_path = 'movies.csv'    

ratings_data = pd.read_csv(ratings_file_path)
movies_data = pd.read_csv(movies_file_path)

movies_data['genres'] = movies_data['genres'].str.replace('|', ' ')

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def movie_recommender_engine(movie_name, n_recs=5):
    closest_match = process.extractOne(movie_name, movies_data['title'])
    
    if not closest_match:
        return "Movie not found."
    
    movie_title = closest_match[0]
    movie_index = movies_data[movies_data['title'] == movie_title].index[0]

    sim_scores = list(enumerate(cosine_sim[movie_index]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_movie_indices = [i[0] for i in sim_scores[1:n_recs + 1]]  
    recommended_movies = movies_data['title'].iloc[top_movie_indices].reset_index(drop=True)
    return recommended_movies

recommended_movies = movie_recommender_engine("The Dark Knight", n_recs=5)

print("Recommended Movies:")
print(recommended_movies)