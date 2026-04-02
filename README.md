# CineMatch AI - Hybrid Movie Recommendation System

CineMatch AI is a capstone-level web application that utilizes both Content-Based and Collaborative Filtering machine learning algorithms to provide highly personalized movie recommendations.

## Features

- **Hybrid Recommendation Engine**: Fuses TF-IDF genre analysis with Truncated SVD (Singular Value Decomposition) on an item-user matrix to solve the Cold Start problem.
- **Large Dataset Support**: Integrates directly with the official MovieLens Latest Small dataset (100,000+ ratings, nearly 10,000 movies).
- **TMDB Posters & Metadata Integration**: Insert a free API key from The Movie Database into the frontend header to map internal IDs and instantly retrieve real-world movie posters.
- **FastAPI Backend**: Rapid algorithm computation and semantic searching via RapidFuzz and scikit-learn.
- **React + Vite Frontend**: Responsive, modern, and beautiful UI with dark/glassmorphic themes.

## Architecture 

- `backend/`: Powered by FastAPI, Pandas, and scikit-learn. Loads datasets recursively into sparse matrices on startup.
- `frontend/`: Powered by React and Vite. Provides the user interface, search parameters, and dynamically fetches from TMDB.

## Installation and Execution

### 1. Backend Setup

Open a terminal and navigate to the backend folder:
```bash
cd backend

# Install dependencies (ensure Python is installed)
pip install -r requirements.txt

# Start the Python inference server manually
uvicorn main:app --reload
```
The backend initializes the dataset into memory matrix arrays on startup, which will take a couple of seconds. You will see `Data loaded and Hybrid models initialized successfully.` when it's ready.

### 2. Frontend Setup

Open a second terminal and navigate to the frontend folder:
```bash
cd frontend

# Install node dependencies
npm install

# Start the Vite development server
npm run dev
```
Navigate to `http://localhost:5173/` in your web browser.

## Using TMDB (Optional but Recommended)

By default, the UI will fall back to using first-letter placeholder icons if it cannot fetch actual images. If you wish to use real images:
1. Create a free developer API key at tmdb.org
2. Open the React site `http://localhost:5173/`
3. Locate the `TMDB API Key` field in the top-right header and paste in your key.

## Data Structure

The required dataset relies on three files in the `/backend/data/` directory:
- `movies.csv` (Standard MovieLens format)
- `ratings.csv` (Standard User-Item rating logic)
- `links.csv` (Standard MovieLens ecosystem bridging IDs, explicitly the `tmdbId` mapping logic)
