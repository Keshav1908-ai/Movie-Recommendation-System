import React, { useState, useEffect } from 'react';
import './App.css';

// Using a premium dark theme approach
function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [recommendedFor, setRecommendedFor] = useState('');
  const [loading, setLoading] = useState(false);

  // Fetch recommendations for a default movie on load
  useEffect(() => {
    fetchRecommendations('The Dark Knight');
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const resp = await fetch(`http://localhost:8000/api/movies/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await resp.json();
      setSearchResults(data.results || []);
      
      // If we find exactly what they searched for, also get recommendations for it
      if (data.results && data.results.length > 0) {
        fetchRecommendations(data.results[0].title);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async (title) => {
    setLoading(true);
    try {
      const resp = await fetch(`http://localhost:8000/api/recommendations?movie_name=${encodeURIComponent(title)}&n_recs=5`);
      const data = await resp.json();
      setRecommendations(data.recommendations || []);
      setRecommendedFor(data.target_movie || '');
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="logo">CineMatch <span>AI</span></h1>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="Search for movies, genres..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="search-btn">Search</button>
        </form>
      </header>

      <main className="main-content">
        {loading && <div className="loader">Analyzing Matrix...</div>}

        {!loading && searchResults.length > 0 && (
          <section className="dashboard-section">
            <h2 className="section-title">Search Results</h2>
            <div className="movie-grid">
              {searchResults.map(movie => (
                <div key={`search-${movie.id}`} className="movie-card" onClick={() => fetchRecommendations(movie.title)}>
                  <div className="movie-poster">
                    {/* Placeholder for real images */}
                    <div className="poster-placeholder">{movie.title.charAt(0)}</div>
                  </div>
                  <div className="movie-info">
                    <h3>{movie.title}</h3>
                    <p className="genres">{movie.genres.split('|').join(' • ')}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {!loading && recommendations.length > 0 && (
          <section className="dashboard-section">
            <h2 className="section-title">Because you like <span className="highlight">{recommendedFor}</span></h2>
            <div className="movie-grid">
              {recommendations.map(movie => (
                <div key={`rec-${movie.id}`} className="movie-card">
                  <div className="movie-poster">
                    <div className="poster-placeholder rec-placeholder">{movie.title.charAt(0)}</div>
                  </div>
                  <div className="movie-info">
                    <h3>{movie.title}</h3>
                    <p className="genres">{movie.genres}</p>
                    <button className="watch-btn">Watch Trailer</button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
