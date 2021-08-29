import streamlit as st

from utils.utils_app_front import add_preview_url, get_footer
from utils.utils_app import load_data_pkl, get_popular_songs_year


if __name__ == '__main__':
    st.image("misc/music_img.jpg")
    st.markdown("<h1 style='text-align: center; color: red;'>"
                "<center>&emsp;&emsp;Music Recommendation System</center></h1>", unsafe_allow_html=True)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    choices = ["choose a recommendation type", "Popular Songs", "Similar Artists Recommendation for Users"]
    st.write("Recommendation Type")
    choice = st.selectbox("", choices, 0)
    st.write(" ")
    st.markdown(get_footer(), unsafe_allow_html=True)

    if choice == "Popular Songs":
        data = load_data_pkl('data/tracks.pkl')
        year = st.slider("Year", min_value=1975, max_value=2021, value=2009, step=1)
        K = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5, step=1)
        popular_songs_year = get_popular_songs_year(data, year)[:K]
        popular_songs_year_html = add_preview_url(popular_songs_year)
        st.markdown(popular_songs_year_html, unsafe_allow_html=True)

    elif choice == "Similar Artists Recommendation for Users":
        rec = load_data_pkl('data/recommendations-artists.pkl')
        st.table(rec.head())
