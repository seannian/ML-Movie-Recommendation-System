# Movie Recommendation System

## Brief Description

This project is a web-based Movie Recommendation System that suggests movies based on user input. It leverages a hybrid approach, combining content-based and collaborative filtering techniques to provide personalized recommendations.

**Key Features:**

- **Movie Search:** Users can search for movies by title.
- **Content-Based Search:** Uses TF-IDF vectorization on movie titles to find the closest matches to the user's search query.
- **Collaborative Filtering Recommendations:** Employs Singular Value Decomposition (SVD) from the Surprise library to predict user ratings and generate movie recommendations.
- **Genre-Aware Recommendations:** Enhances collaborative filtering by prioritizing recommendations within similar genres to the searched movie.
- **Web Interface:** Built with Flask and Bootstrap for a user-friendly and responsive experience.
- **Movie Details:** Fetches movie posters, release dates, ratings, and overviews from the TMDB API to display rich movie information.

## Technologies Used

- **Python:** Programming language for the backend logic and Flask application.
- **Flask:** Web framework for building the web application.
- **Bootstrap 5:** CSS framework for styling the web interface and ensuring responsiveness.
- **Pandas:** Data manipulation and analysis library for handling movie datasets.
- **scikit-learn:** Machine learning library, used for TF-IDF vectorization and cosine similarity.
- **scikit-surprise (Surprise):** Recommender systems library, used for implementing the SVD collaborative filtering model.
- **requests:** HTTP library for making requests to the TMDB API.
- **Pickle:** Python module for serializing and de-serializing Python object structures (used for saving and loading models and data).
- **HTML/CSS/JavaScript:** For the frontend structure and basic interactivity (managed by Flask and Bootstrap).
- 
## Installation Instructions

Follow these steps to set up and run the Movie Recommendation System locally:

**Prerequisites:**

- **Python 3.10 or higher:** It is recommended to use Python 3.10 as indicated in the project's `pyproject.toml`. You can check your Python version by running `python --version` in your terminal.
- **Poetry:** This project uses Poetry for dependency management. If you don't have Poetry installed, follow the [official installation guide](https://python-poetry.org/docs/#installation).

**Steps:**

1. **Clone the repository:**
   ```bash
   git clone <repository_url>  # Replace <repository_url> with the actual repository URL
   cd ML-Movie-Recommendation-System-main
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```
   This command will create a virtual environment and install all necessary Python packages listed in `pyproject.toml`.

   Alternatively, if you don't want to use Poetry, you can use `pip`. First, ensure you are in the project directory and then run:
   ```bash
   pip install -r requirements.txt
   pip install scikit-surprise
   pip install flask flask-bootstrap requests
   ```

3. **Obtain a TMDB API Key:**
   - To fetch movie details and posters, you need an API key from [The Movie Database (TMDB)](https://www.themoviedb.org/).
   - Sign up for a free account on TMDB.
   - Go to your account settings and request an API key.

4.  **Create a Configuration File (`config.py`) and Set TMDB API Key:**
    - In the root directory of your project, create a new Python file named `config.py`.
    - Inside `config.py`, define a variable to store your TMDB API key:

        ```python
        TMDB_API_KEY = "YOUR_ACTUAL_TMDB_API_KEY_HERE"  # Replace with your API key
        ```

    - **Important:**  Replace `"YOUR_ACTUAL_TMDB_API_KEY_HERE"` with your *actual* TMDB API key.
    - **Security Note:** This `config.py` file is intentionally *not* tracked by Git (see step 5). This prevents your API key from being accidentally committed to version control.

5.  **Ensure `config.py` is Ignored by Git:**
    - Make sure you have a `.gitignore` file in the root of your project.
    - Verify that `.gitignore` contains the line `config.py` (or whatever you named your configuration file). This tells Git to ignore this file and prevent it from being committed.

6. **Run the Flask application:**
   ```bash
   python app.py
   ```
   This will start the Flask development server. You should see an output like:
   ```
   * Serving Flask app 'app'
   * Debug mode: on
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
   * Running on http://127.0.0.1:5000
   Press CTRL+C to quit
   * Restarting with stat
   * Debugger is active!
   * Debugger PIN: ...
   ```

7. **Access the application in your browser:**
   - Open your web browser and go to `http://127.0.0.1:5000` or `http://localhost:5000`.

## Usage Guide

1. **Movie Search:**
   - Once the application is running in your browser, you will see a "Movie Search" section with an input field labeled "Movie Title".
   - Enter the title of a movie you are interested in.
   - Click the "Submit" button.

2. **Movie Recommendations:**
   - After submitting a movie title, the system will process your request and redirect you to a "Recommended Movies" page.
   - This page displays a list of movies recommended based on your search query and the underlying recommendation algorithms.
   - For each recommended movie, you will see:
     - Movie Poster
     - Movie Title (in bold)
     - Release Date (in italics)
     - Average Rating from TMDB (in italics)
     - Movie Overview/Synopsis

3. **Exploring Recommendations:**
   - Scroll through the list of recommended movies to discover new movies you might enjoy.
   - The recommendations are genre-aware and are based on collaborative filtering, aiming to provide personalized suggestions.

---
