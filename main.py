from emotion_detector import detect_emotion
from genre_mapper import get_genre
from spotify_recommender import get_tracks_by_genre, show_tracks

print("🎥 Detecting Emotion...")
emotion = detect_emotion()
print(f"Detected Emotion: {emotion}")

genre = get_genre(emotion)
print(f"🎵 Recommended Genre: {genre}")

tracks = get_tracks_by_genre(genre)
show_tracks(tracks)
