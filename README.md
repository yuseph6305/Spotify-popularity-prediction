# Predicting Spotify Song Popularity with Machine Learning

A portfolio-ready regression project that predicts a Spotify track's **popularity** score from audio features like `danceability`, `energy`, `valence`, `tempo`, etc.

## Why this is internship-ready
- Clear, reproducible **Jupyter workflow** (EDA → modeling → evaluation)
- Multiple **regression models** with tuned hyperparameters
- Clean project structure and **README** documentation
- Optional **Streamlit app** for interactive demos
- Optional **Spotify API** integration for live lookups

## Dataset
Use either:
1. **Kaggle**: *Ultimate Spotify Tracks DB* (CSV) — place the file at `data/spotify_tracks.csv`
2. **Spotify Web API** via `spotipy` — collect your own dataset with `src/collect_spotify_api.py` (to add later if desired)

> Expected columns include: `danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, popularity, year, explicit, mode, time_signature, artist, track_name, track_id` (columns beyond features will be dropped in modeling).

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Put the dataset in place
#    Save your CSV as: data/spotify_tracks.csv

# 4) Open the notebook
jupyter lab  # or jupyter notebook
# Run: notebooks/01_spotify_popularity.ipynb

# 5) (Optional) Train & persist from CLI
python src/train.py --data data/spotify_tracks.csv --model-out models/final_model.pkl
python src/evaluate.py --data data/spotify_tracks.csv --model models/final_model.pkl

# 6) (Optional) Run the Streamlit app
streamlit run app/streamlit_app.py
```

## Project Structure
```
spotify_popularity_prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   └── spotify_tracks.csv           # <- put your dataset here
├── models/
│   └── final_model.pkl              # saved after training
├── notebooks/
│   └── 01_spotify_popularity.ipynb  # full EDA→modeling notebook
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Notes
- The notebook uses `scikit-learn` pipelines and `GridSearchCV` to compare: **LinearRegression, Ridge, Lasso, RandomForestRegressor, XGBRegressor** (if available).
- SHAP-based interpretability is included if you choose to install `shap`.
- The Streamlit app works even **without** Spotify credentials by allowing manual feature input; with credentials, it can auto-fetch song features.
# Spotify-popularity-prediction
