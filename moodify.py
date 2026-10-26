import os
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from urllib.parse import quote_plus


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
# 2. Map emotions to music genres
# =========================================================

def get_genre(emotion):
    mapping = {
        "Happy": "happy pop music",
        "Sad": "sad acoustic songs",
        "Angry": "rock music",
        "Surprise": "energetic dance music",
        "Neutral": "chill relaxing music",
        "Fear": "calming ambient music",
        "Disgust": "heavy metal music"
    }

    return mapping.get(emotion, "popular music")


# =========================================================
# 3. Generate YouTube search links
# =========================================================

def get_youtube_tracks(genre):
    recommendations = [
        f"{genre} playlist",
        f"best {genre} songs",
        f"{genre} music mix",
        f"{genre} songs with lyrics",
        f"popular {genre}"
    ]

    tracks = []

    for query in recommendations:
        tracks.append({
            "name": query.title(),
            "url": (
                "https://www.youtube.com/results?search_query="
                + quote_plus(query)
            )
        })

    return tracks


# =========================================================
# 4. Detect emotion from face
# =========================================================

def detect_emotion_from_image(image):
    if image is None:
        return "No image received"

    input_shape = emotion_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported model input shape: {input_shape}"
        )

    height = input_shape[1] or 48
    width = input_shape[2] or 48
    channels = input_shape[3]

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported number of input channels: {channels}"
        )

    # Convert camera image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load face detector.")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return "No face detected"

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (int(width), int(height)))

    # Prepare input according to model channel count
    if channels == 1:
        roi = face[..., np.newaxis]
    else:
        roi = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    # Match the preprocessing used during model training
    roi = roi.astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=0)

    prediction = emotion_model.predict(roi, verbose=0)
    probabilities = prediction[0]

    if len(probabilities) != len(emotion_labels):
        raise ValueError(
            "Model output count does not match emotion labels."
        )

    emotion_index = int(np.argmax(probabilities))
    emotion = emotion_labels[emotion_index]
    confidence = float(probabilities[emotion_index])

    return emotion, confidence, image, (x, y, w, h)


# =========================================================
# 5. Streamlit application
# =========================================================

def main():
    st.set_page_config(
        page_title="Moodify - YouTube Music",
        page_icon="🎵",
        layout="centered"
    )

    st.title("🎵 Moodify")
    st.subheader("Facial Emotion-Based Music Recommendation")

    st.write(
        "Capture your face to estimate your facial expression "
        "and explore music recommendations on YouTube."
    )

    img_file = st.camera_input("📸 Capture your face")

    if img_file is None:
        st.info("Capture a photo to get started.")
        return

    # Decode camera image
    file_bytes = np.frombuffer(
        img_file.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Unable to read the image. Please try again.")
        return

    # Predict emotion
    try:
        result = detect_emotion_from_image(frame)

        if isinstance(result, str):
            st.warning(result)
            return

        emotion, confidence, original_image, face_box = result

    except Exception as error:
        st.error("Emotion detection failed.")
        st.exception(error)
        return

    # Draw face rectangle
    x, y, w, h = face_box
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
        "This score is not necessarily a calibrated probability "
        "or a definitive measure of your mood."
    )

    # Generate music recommendations
    genre = get_genre(emotion)

    st.subheader(f"🎶 Recommended music: {genre.title()}")
    st.write("Choose a search below to explore music on YouTube.")

    recommendations = get_youtube_tracks(genre)

    for idx, track in enumerate(recommendations, start=1):
        st.markdown(f"**{idx}. {track['name']}**")
        st.markdown(f"[▶️ Listen on YouTube]({track['url']})")
        st.divider()


# =========================================================
# 6. Run application
# =========================================================

if __name__ == "__main__":
    main()
