import requests
import pandas as pd
from credentials import SPOTIFY_TOKEN

pd.set_option('colheader_justify', 'center')
pd.set_option('display.width', 1000)


def preprocess(data):
    nonvalid_row_idx = data[data.release_date == "1900-01-01"].index
    processed_data = data.drop(nonvalid_row_idx, axis=0)
    processed_data['year'] = pd.DatetimeIndex(processed_data['release_date']).year
    processed_data.artists = processed_data.artists.apply(lambda x: x[1:-1].replace("'", ""))
    processed_data.rename(columns={"name": "Song Name",
                                   "artists": "Artist",
                                   "year": "Release Year",
                                   "popularity": "Popularity"}, inplace=True)
    processed_data = processed_data[['Song Name', 'Artist', 'Release Year', 'Popularity', 'id']]
    return processed_data


def get_img_and_track_url(id):
    query = f"https://api.spotify.com/v1/tracks/{id}?market=US"
    response = requests.get(
        query,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(SPOTIFY_TOKEN)
        },
        verify = False
    )
    response_json = response.json()
    return response_json['album']['images'][1]['url'], response_json['external_urls']['spotify']

