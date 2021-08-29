import pandas as pd
import streamlit as st

from utils.utils import preprocess
from rec import get_popular_songs


@st.cache(show_spinner=False)
def load_data_pkl(path):
    data = pd.read_pickle(path)
    return data

@st.cache(show_spinner=False)
def get_popular_songs_year(data, year):
    processed_data = preprocess(data)
    popular_songs = get_popular_songs(processed_data, year)
    popular_songs.reset_index(drop=True, inplace=True)
    popular_songs.index += 1
    return popular_songs
