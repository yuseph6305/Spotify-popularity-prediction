import pandas as pd
import numpy as np

NUMERIC_FEATURES = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','duration_ms'
]

TARGET_COL = 'popularity'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only numeric features + target
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if TARGET_COL in df.columns:
        cols = cols + [TARGET_COL]
    out = df[cols].copy()
    # Drop rows with missing target or all-nan features
    out = out.dropna(subset=[TARGET_COL])
    out = out.dropna()
    # Remove extreme outliers in popularity if present
    out = out[(out[TARGET_COL] >= 0) & (out[TARGET_COL] <= 100)]
    # Clip tempo & loudness to reasonable ranges
    if 'tempo' in out.columns:
        out['tempo'] = out['tempo'].clip(lower=30, upper=220)
    if 'loudness' in out.columns:
        out['loudness'] = out['loudness'].clip(lower=-60, upper=5)
    return out
