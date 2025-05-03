from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = FastAPI(title="Recommendation Service")

def load_song_metadata():
    data = [
        {"song_id": 1, "title": "Blinding Lights", "genre": "pop"},
        {"song_id": 2, "title": "Levitating", "genre": "pop"},
        {"song_id": 3, "title": "Watermelon Sugar", "genre": "pop"},
        {"song_id": 4, "title": "Circles", "genre": "rock"},
        {"song_id": 5, "title": "Don’t Start Now", "genre": "dance"},
        {"song_id": 6, "title": "Rockstar", "genre": "hip hop"},
        {"song_id": 7, "title": "Savage Love", "genre": "pop"},
        {"song_id": 8, "title": "Sunflower", "genre": "rap"},
        {"song_id": 9, "title": "Memories", "genre": "pop"},
        {"song_id": 10, "title": "Mood", "genre": "rap"}
    ]
    return pd.DataFrame(data)

def load_user_likes():
    return {
        1: [1, 3, 9],  # user 1 likes songs 1,3,9
        2: [4, 6],     # user 2 likes songs 4,6
        3: [2, 5, 7]   # user 3 likes songs 2,5,7
    }

song_df = load_song_metadata()
song_df['text'] = song_df['title'] + ' ' + song_df['genre']
vectorizer = TfidfVectorizer()
song_embeddings = vectorizer.fit_transform(song_df['text'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(song_embeddings)

user_likes = load_user_likes()

class Recommendation(BaseModel):
    song_ids: list[int]

@app.get("/recommendations/{user_id}", response_model=Recommendation)
def get_recommendations(user_id: int, limit: int = 5):
    if user_id not in user_likes:
        raise HTTPException(status_code=404, detail="User not found")
    liked = user_likes[user_id]
    liked_indices = [song_df.index[song_df['song_id'] == sid][0] for sid in liked]
    user_emb = song_embeddings[liked_indices].mean(axis=0)
    distances, indices = nn_model.kneighbors(user_emb, n_neighbors=limit + len(liked))
    rec_indices = [i for i in indices.flatten() if song_df.at[i, 'song_id'] not in liked]
    rec_indices = rec_indices[:limit]
    rec_ids = [int(song_df.at[i, 'song_id']) for i in rec_indices]
    return Recommendation(song_ids=rec_ids)
