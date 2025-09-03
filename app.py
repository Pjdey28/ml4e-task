
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SongRecommender:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None
        self.scaler = None
        self.df = None
        self.feature_cols = None

    def preprocess(self, df, feature_cols):
        """Save dataset and fit scaler"""
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[feature_cols])
        return X_scaled

    def fit(self, df, feature_cols):
        """Fit kNN model on scaled features"""
        X_scaled = self.preprocess(df, feature_cols)
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors+1, metric="cosine"
        )
        self.model.fit(X_scaled)

    def recommend(self, song_index):
        """Return recommended song indices"""
        X_scaled = self.scaler.transform(self.df[self.feature_cols])
        distances, indices = self.model.kneighbors([X_scaled[song_index]])
        recs = indices[0][1:]
        return self.df.iloc[recs][["track_name", "artist_name", "genre"]]

     

import streamlit as st
from recommender import SongRecommender

df = pd.read_csv("./tcc_ceds_music.csv")

feature_cols = ["valence", "energy", "danceability", "acousticness", "instrumentalness"]

rec = SongRecommender(n_neighbors=5)
rec.fit(df, feature_cols)

st.title("Song Recommendendation system")

song_list = df["track_name"].dropna().unique().tolist()
selected_song = st.selectbox("Choose a song:", song_list)

if st.button("Recommend Similar Songs"):
    idx = df[df["track_name"] == selected_song].index[0]
    recs = rec.recommend(idx)
    st.write("### Recommended Songs")
    st.dataframe(recs)

     
