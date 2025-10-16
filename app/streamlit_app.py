import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

FEATURES = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','duration_ms'
]

st.set_page_config(page_title='Spotify Popularity Predictor', page_icon='🎧')
st.title('🎧 Spotify Song Popularity Predictor')
st.write('Predict a track\'s Spotify *popularity* (0–100) from its audio features.')

model_path = os.environ.get('MODEL_PATH', 'models/final_model.pkl')
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success('Loaded trained model.')
else:
    st.warning('Trained model not found. Train one via the notebook or src/train.py.')

st.header('Enter Features')
cols = st.columns(2)
inputs = {}
with cols[0]:
    inputs['danceability'] = st.slider('danceability', 0.0, 1.0, 0.5, 0.01)
    inputs['energy'] = st.slider('energy', 0.0, 1.0, 0.6, 0.01)
    inputs['loudness'] = st.slider('loudness (dB)', -60.0, 5.0, -7.0, 0.1)
    inputs['speechiness'] = st.slider('speechiness', 0.0, 1.0, 0.05, 0.01)
    inputs['acousticness'] = st.slider('acousticness', 0.0, 1.0, 0.2, 0.01)
with cols[1]:
    inputs['instrumentalness'] = st.slider('instrumentalness', 0.0, 1.0, 0.0, 0.01)
    inputs['liveness'] = st.slider('liveness', 0.0, 1.0, 0.15, 0.01)
    inputs['valence'] = st.slider('valence', 0.0, 1.0, 0.5, 0.01)
    inputs['tempo'] = st.slider('tempo (BPM)', 30.0, 220.0, 120.0, 1.0)
    inputs['duration_ms'] = st.number_input('duration (ms)', min_value=10000, max_value=600000, value=210000, step=1000)

df = pd.DataFrame([inputs])

if st.button('Predict Popularity'):
    if model is None:
        st.error('Please train a model first.')
    else:
        pred = model.predict(df)[0]
        st.metric('Predicted Popularity', f"{pred:.1f}")        
        st.caption('Tip: Train with more diverse data for better generalization.')
