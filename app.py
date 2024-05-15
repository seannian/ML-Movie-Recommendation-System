from pprint import pprint
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from flask_bootstrap import Bootstrap5
from requests import get

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import ipywidgets as widgets
import pickle
import re

algo = pickle.load(open('model.pkl', 'rb'))
df = pd.read_pickle("./df.pkl")
movies = pd.read_pickle("./movies.pkl")
links = pd.read_pickle("./links.pkl")
ratings = pd.read_pickle("./ratings.pkl")
tags = pd.read_pickle("./tags.pkl")
with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('tfidf.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)

app = Flask(__name__)
bootstrap = Bootstrap5(app)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


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


def search(title):
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results


def get_top_n_recommendations(model, movie_id, n=10):
    # Find similar movies to the given movie ID
    similar_movie_ids = find_similar_movies_cf_with_genre(movie_id)
    # Predict ratings for similar movies
    predictions = []
    for movie_id in similar_movie_ids:
        prediction = model.predict(uid='dummy_user', iid=movie_id)
        predictions.append(prediction)


    # Sort predictions by estimated rating in descending order
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Retrieve movie names using their ids
    top_n_movie_titles = []
    for prediction in top_predictions[:n]:
        movie_id = prediction.iid
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        top_n_movie_titles.append(movie_title)

    return top_n_movie_titles


def find_similar_movies_cf_with_genre(movie_id, n=10):
    # Get genre information for the input movie
    input_movie_genres = movies[movies['movieId'] ==
                                movie_id]['genres'].iloc[0]

    # Filter similar movies by genre
    filtered_similar_movie_ids = []
    for movie_id in movies["movieId"]:
        movie_genres = movies[movies['movieId'] == movie_id]['genres'].iloc[0]
        if input_movie_genres in movie_genres:
            filtered_similar_movie_ids.append(movie_id)
    return filtered_similar_movie_ids
