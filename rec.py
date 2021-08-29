
def recommend_user(recommendations, user_id):
    recommendations_user = recommendations[recommendations['user_id'] == user_id]
    return recommendations_user

def get_popular_songs(processed_data, year):
    processed_data_year = processed_data[processed_data['Release Year'] == year]
    popular_songs_year = processed_data_year.sort_values(by=["Popularity"], ascending=False)[:50]
    return popular_songs_year.drop(columns=['Popularity'])
