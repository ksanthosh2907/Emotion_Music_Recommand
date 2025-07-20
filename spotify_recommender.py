import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser

client_id = "4d2013be9e5540e9bed0ad45842c4dc1"
client_secret = "25ba0bac19004e68be11e0a0e9fa6eb2"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id, client_secret))

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

def show_tracks(tracks):
    for idx, track in enumerate(tracks):
        print(f"{idx + 1}. {track['name']} by {track['artist']}")
        print(f"   {track['url']}")
    webbrowser.open(tracks[0]['url'])  # play first song
