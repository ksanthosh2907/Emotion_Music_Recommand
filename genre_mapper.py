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
