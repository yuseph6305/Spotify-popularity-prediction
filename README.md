Predicting Spotify Song Popularity with Machine Learning

This project uses machine learning to predict a song’s popularity score on Spotify based on its audio features such as danceability, energy, valence, and tempo. I wanted to explore what makes certain songs more popular than others and see how well data can capture musical trends.

Project Overview

I used data from Spotify’s public tracks database on Kaggle and tested several regression models to predict a song’s popularity. The project includes data cleaning, exploration, model building, and evaluation in a Jupyter notebook.

There’s also an optional Streamlit app that lets you adjust song features and instantly see the predicted popularity score.

Dataset

You can use one of the following:

Kaggle: Ultimate Spotify Tracks DB
 — place the file in data/spotify_tracks.csv

Spotify Web API: You can also collect your own dataset using the spotipy Python library (this is optional).

The dataset includes features like:
danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, and popularity.

How to Run It
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt

# 3. Place your dataset in the data folder
# Example:
# data/spotify_tracks.csv

# 4. Open the Jupyter notebook
jupyter lab
# or
jupyter notebook

# 5. Run the notebook step-by-step
# notebooks/01_spotify_popularity.ipynb


You can also train and evaluate models directly from the command line:

python src/train.py --data data/spotify_tracks.csv --model-out models/final_model.pkl
python src/evaluate.py --data data/spotify_tracks.csv --model models/final_model.pkl


To try the Streamlit web app:

streamlit run app/streamlit_app.py

Project Structure
spotify_popularity_prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   └── spotify_tracks.csv
├── models/
│   └── final_model.pkl
├── notebooks/
│   └── 01_spotify_popularity.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
└── README.md

Notes

The project compares several regression models: Linear Regression, Ridge, Lasso, Random Forest, and XGBoost.

GridSearchCV is used to tune hyperparameters.

SHAP interpretability is supported if installed.

The Streamlit app works without Spotify API credentials, but you can connect the API to fetch live song features.

Reflection

This project helped me practice real-world data science steps: cleaning and preparing data, building regression models, and interpreting results. It was interesting to see how audio features like energy, valence, and danceability affect a song’s popularity.
