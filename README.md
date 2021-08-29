# music_recommendation_system

## About:
I built a music recommendation system that comprises 2 features:<br>
* **Popular Music Recommendation:** recommend to the user the most popular songs according to the selected year <br>
* **Similar Artists Recommendation:** recommend to the user artists based on his listening history.<br>
This approach is called collaborative filtering and it takes advantage of an optimization algorithm named ALS (alternating least square) where the number of times an artist was played by an user is considered as an implicit feedback.
(details about this approach can be found in this [paper](http://yifanhu.net/PUB/cf.pdf)

## Demo video:
(link to video)

## Technologies:
* [Databricks](https://databricks.com/try-databricks)
* [Pyspark](http://spark.apache.org/docs/latest/api/python/)
* [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
* [Streamlit](https://streamlit.io/)

## Recommendation Model Training:

All the conducted steps (data loading, data processing, data exploration, model training, and model evaluation) are illustrated in the following Databricks [notebook]( https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/550853209436599/750566902736418/4358824653449923/latest.html)


The data set was split into training and testing sets then the resulting model was evaluated on the test set via mean average precision @10. The score that was reported is 0.96

## Datasets (data folder):

Similar Artists Recommendation feature: [Last.fm dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html)<br>
Popular Music Recommendation feature:  dataset previously available on Kaggle, refer to the [data folder](https://drive.google.com/drive/folders/1AQzrv0q2cwRXMYyHRbsE6zFF1Sr7WQOx?usp=sharing)
