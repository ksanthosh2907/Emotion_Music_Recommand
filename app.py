import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import webbrowser
import tempfile
import os

# Load emotion model
emotion_model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify auth
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id, client_secret))

# Emotion → Genre
def get_genre(emotion):
    mapping = {
        'Happy': 'pop',
        'Sad': 'acoustic',
        'Angry': 'rock',
        'Surprise': 'dance',
        'Neutral': 'chill',
        'Fear': 'ambient',
        'Disgust': 'metal'
    }
    return mapping.get(emotion, 'pop')

# Spotify fetch
def get_tracks_by_genre(genre):
    results = sp.search(q=f'genre:{genre}', type='track', limit=5)
    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'url': item['external_urls']['spotify']
        })
    return tracks

def detect_emotion_from_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi[np.newaxis, :, :, np.newaxis] / 255.0
        prediction = emotion_model.predict(roi)
        max_index = int(np.argmax(prediction))
        return emotion_labels[max_index]
    return "Neutral"

# Streamlit UI
st.title("🎵 Music Recommendation from Facial Emotion")

img_file = st.camera_input("Capture your face")

if img_file is not None:
    # Save and read image
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(img_file.getvalue())
    frame = cv2.imread(tfile.name)

    emotion = detect_emotion_from_image(frame)
    genre = get_genre(emotion)
    tracks = get_tracks_by_genre(genre)

    st.subheader(f"Detected Emotion: 😄 {emotion}")
    st.subheader(f"Recommended Genre: 🎶 {genre}")

    for idx, track in enumerate(tracks):
        st.markdown(f"**{idx + 1}. {track['name']}** by *{track['artist']}*")
        st.markdown(f"[🔗 Listen on Spotify]({track['url']})")

    # Auto open first track
    webbrowser.open(tracks[0]['url'])
