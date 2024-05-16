import numpy as np
import pandas as pd
import pickle
import re

from flask import Flask, redirect, render_template, request, session, url_for
from flask_bootstrap import Bootstrap5
from requests import get
from sklearn.metrics.pairwise import cosine_similarity

algo = pickle.load(open('pickles/model.pkl', 'rb'))
df = pd.read_pickle("pickles/df.pkl")
movies = pd.read_pickle("pickles/movies.pkl")
links = pd.read_pickle("pickles/links.pkl")
ratings = pd.read_pickle("pickles/ratings.pkl")
tags = pd.read_pickle("pickles/tags.pkl")
with open('pickles/vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('pickles/tfidf.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)

app = Flask(__name__)
bootstrap = Bootstrap5(app)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')

# Handle submit requests
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        sometext = request.form['inputText']
        results = search(sometext)
        if not results.empty:
            movie_id = results.iloc[0]["movieId"]
            session['movieArray'] = [
                re.sub(r'\s*\([^)]*\)', '', movie).rstrip()
                for movie in get_top_n_recommendations(algo, movie_id)
            ]
        return redirect(url_for('display_movies'))

# Display Movies using loop
@app.route('/display_movies')
def display_movies():
    movieData = []

    for movie in session.get('movieArray', []):
        response = get(
            f'https://api.themoviedb.org/3/search/movie?query={movie}&api_key=e2b56f824d3987e10f41f792247e32ec'
        ).json()
        if not response['results']:
            continue
        searchedMovie = response['results'][0]
        searchedMovie[
            'poster'] = f'http://image.tmdb.org/t/p/w500{searchedMovie["poster_path"]}'
        movieData.append(searchedMovie)
    return render_template('movies.html', movies=movieData)


if __name__ == "__main__":
    app.run(debug=True)


# find movie given the input
def search(title):
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results


# Given a movie id, return top 10 recommendations
def get_top_n_recommendations(model, movie_id, n=10):

    similar_movie_ids = find_similar_movies_cf_with_genre(movie_id)
    # Predict ratings for similar movies
    predictions = []
    for movie_id in similar_movie_ids:
        prediction = model.predict(uid='dummy_user', iid=movie_id)
        predictions.append(prediction)

    # Sort predictions from highest to lowest
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Retrieve movie names using their ids
    top_n_movie_titles = []
    for prediction in top_predictions[:n]:
        movie_id = prediction.iid
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        top_n_movie_titles.append(movie_title)

    return top_n_movie_titles


# Find similar movies based on movie genre
def find_similar_movies_cf_with_genre(movie_id, n=10):

    input_movie_genres = movies[movies['movieId'] ==
                                movie_id]['genres'].iloc[0]

    # Filter by genre
    filtered_similar_movie_ids = []
    for movie_id in movies["movieId"]:
        movie_genres = movies[movies['movieId'] == movie_id]['genres'].iloc[0]
        if input_movie_genres in movie_genres:
            filtered_similar_movie_ids.append(movie_id)
    return filtered_similar_movie_ids
