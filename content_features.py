import pandas as pd
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

LAST_FM_API_KEY = os.getenv('LASTFM_API_KEY')
ACOUSTICBRAINZ_API_KEY = os.getenv('ACOUSTICBRAINZ_API_KEY')
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}

# build content based filtering database using freshly generated collaborative filtering database 
def find_most_played():
    recent_tracks = pd.read_csv("lastfm_recent_tracks.csv")
    columns = ['artist', 'name', 'album']
    mbid_columns = ['artist_mbid', 'name_mbid','album_mbid']
    song_counts = recent_tracks.groupby(columns).size().reset_index(name='count')
    # go through every unique song and find every possible mbids 
    mbid_counts = (recent_tracks.groupby(columns)[mbid_columns]
    .agg(lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) > 0 else None)
    .reset_index()
    ) 
    # merge count and mbid dataframes
    counted_dataset= pd.merge(song_counts,mbid_counts,on=columns).sort_values('count',ascending=False)

    content_dataset = counted_dataset.head(250000).copy()
    content_dataset.to_csv("lastfm_most_played_250k.csv")

# get last.fm tags and acousticbrainz tags for every song
# should add tags for artist and for the album

def get_lastfm_tags(name, artist):
    # try:
    base_url = "http://ws.audioscrobbler.com/2.0/?method=track.getTopTags"
    params = {
        "method": "track.getTopTags",
        "track": name,
        "artist": artist,
        "api_key": LAST_FM_API_KEY,
        "format": "json"
    }
    try:
        response = requests.get(base_url, params=params,  headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            tags = [tag['name'] for tag in data.get('toptags', {}).get('tag',[])][:10]
            return tags
    except Exception as e:
        print(f"error: {str(e)}")
        return None
    return []

def get_acousticbrainz_data(mbid):
    url = f"https://acousticbrainz.org/api/v1/{mbid}/high-level"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(e)

def process_row(row):
    last_fm_tags = get_lastfm_tags(name=row["name"],artist=row["artist"])
    acousticbrainz_data = get_acousticbrainz_data(row["name_mbid"] if row["name_mbid"] else {})
    time.sleep(0.25)
    return {
        "artist": row["artist"],
        "name": row["name"],
        "tags": last_fm_tags,
        "acoustic_features": acousticbrainz_data,
        "mbid": row["name_mbid"]
    }

def create_content_dataset(base):
    features = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_row, row) for _, row in base.iterrows()]

        for i, future in enumerate(futures):
            try:
                result = future.result()
                features.append(result)
                print(f"completed {i+1}/{len(base)}")
            except Exception as e:
                print(f"error {str(e)}")
    return pd.DataFrame(features)
#train the dataset 

# find_most_played()
data = pd.read_csv("lastfm_most_played_250k.csv")
content_features = create_content_dataset(data)
content_features.to_csv('content_features.csv', index=False)
# trzeba dodac czas i panstwo