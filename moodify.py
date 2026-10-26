
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import spotipy
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials


# =========================================================
# 1. Load the emotion detection model
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return load_model(MODEL_PATH, compile=False)


emotion_model = load_emotion_model()

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# =========================================================
# 2. Configure Spotify
# =========================================================

@st.cache_resource
def initialize_spotify():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )

    return spotipy.Spotify(auth_manager=auth_manager)


# =========================================================
# 3. Map emotions to music genres
# =========================================================

def get_genre(emotion):
    mapping = {
        "Happy": "pop",
        "Sad": "acoustic",
        "Angry": "rock",
        "Surprise": "dance",
        "Neutral": "chill",
        "Fear": "ambient",
        "Disgust": "metal"
    }

    return mapping.get(emotion, "pop")


# =========================================================
# 4. Fetch songs from Spotify
# =========================================================

def get_tracks_by_genre(genre):
    sp = initialize_spotify()

    results = sp.search(
        q=f"genre:{genre}",
        type="track",
        limit=5
    )

    tracks = []

    for item in results.get("tracks", {}).get("items", []):
        artists = item.get("artists", [])
        external_urls = item.get("external_urls", {})

        tracks.append({
            "name": item.get("name", "Unknown track"),
            "artist": (
                artists[0].get("name", "Unknown artist")
                if artists else "Unknown artist"
            ),
            "url": external_urls.get("spotify", "")
        })

    return tracks


# =========================================================
# 5. Detect face and predict emotion
# =========================================================

def detect_emotion_from_image(image):
    if image is None:
        return "No image received"

    # Determine the input format expected by the model.
    input_shape = emotion_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Expected a 4D image input, but model input is {input_shape}"
        )

    # This code supports channels-last grayscale or RGB models.
    height = input_shape[1] or 48
    width = input_shape[2] or 48
    channels = input_shape[3]

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported model input shape: {input_shape}"
        )

    # Convert camera image (BGR) to grayscale for face detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV face detector.")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return "No face detected"

    # Select the largest face in the image.
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face_gray = gray[y:y + h, x:x + w]

    # Resize to the model's expected dimensions.
    face_gray = cv2.resize(
        face_gray,
        (int(width), int(height))
    )

    # Prepare grayscale or RGB input according to model shape.
    if channels == 1:
        roi = face_gray[..., np.newaxis]
    else:
        roi = cv2.cvtColor(
            face_gray,
            cv2.COLOR_GRAY2RGB
        )

    # Match the common FER2013 pixel normalization.
    # If training used a different normalization, reproduce it here.
    roi = roi.astype(np.float32) / 255.0

    # Add the batch dimension.
    roi = np.expand_dims(roi, axis=0)

    # Predict emotion.
    prediction = emotion_model.predict(roi, verbose=0)
    probabilities = prediction[0]

    if len(probabilities) != len(emotion_labels):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs, "
            f"but {len(emotion_labels)} emotion labels are configured. "
            "Check the model's original class order."
        )

    emotion_index = int(np.argmax(probabilities))
    emotion = emotion_labels[emotion_index]
    confidence = float(probabilities[emotion_index])

    return emotion, confidence, image, (x, y, w, h)


# =========================================================
# 6. Streamlit application
# =========================================================

def main():
    st.set_page_config(
        page_title="Moodify - Emotion Music",
        page_icon="🎵",
        layout="centered"
    )

    st.title("🎵 Moodify")
    st.subheader("Facial Emotion Music Recommendation")

    st.write(
        "Capture your face to estimate your facial expression "
        "and discover music recommendations."
    )

    img_file = st.camera_input("📸 Capture your face")

    if img_file is None:
        st.info("Capture a photo to get started.")
        return

    # Read the uploaded camera image directly from memory.
    file_bytes = np.frombuffer(
        img_file.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Could not read the captured image. Please try again.")
        return

    try:
        result = detect_emotion_from_image(frame)

        if isinstance(result, str):
            st.warning(result)
            return

        emotion, confidence, original_image, face_box = result
        x, y, w, h = face_box

        # Display the captured image with a face rectangle.
        display_image = original_image.copy()

        cv2.rectangle(
            display_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        display_image = cv2.cvtColor(
            display_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_image,
            caption="Detected face",
            use_container_width=True
        )

        st.success(f"Estimated facial emotion: {emotion}")
        st.caption(
            f"Model confidence: {confidence:.1%}. "
            "This is a model score, not a validated measure of actual mood."
        )

    except Exception as error:
        st.error("Emotion detection failed.")
        st.exception(error)
        return

    # Recommend music for the predicted expression.
    genre = get_genre(emotion)

    st.subheader(f"🎶 Recommended genre: {genre.title()}")

    try:
        with st.spinner("Searching Spotify for tracks..."):
            tracks = get_tracks_by_genre(genre)

        if not tracks:
            st.warning(
                "Spotify returned no tracks for this genre. "
                "Please try again later."
            )
            return

        for idx, track in enumerate(tracks, start=1):
            st.markdown(
                f"**{idx}. {track['name']}** — {track['artist']}"
            )

            if track["url"]:
                st.markdown(
                    f"[🎧 Listen on Spotify]({track['url']})"
                )

            st.divider()

    except Exception:
        st.error(
            "Unable to retrieve music from Spotify. "
            "Check your Spotify credentials and API access."
        )


# =========================================================
# 7. Run the application
# =========================================================

if __name__ == "__main__":
    main()
