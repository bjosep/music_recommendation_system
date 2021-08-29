# notebook link: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/550853209436599/750566902736418/4358824653449923/latest.html
# Databricks notebook source
# MAGIC %md
# MAGIC Author: Youssef Belyazid <br>
# MAGIC Email: belyazidyous@gmail.com <br>
# MAGIC Year: 2021

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading lastfm Dataset

# COMMAND ----------

dbutils.fs.mv("dbfs:/FileStore/tables/data_lastfm/data_lastfm.zip","file:/databricks/driver/data_lastfm.zip")

# COMMAND ----------

# MAGIC %%bash
# MAGIC unzip data_lastfm.zip

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/data_lastfm.csv", "dbfs:/FileStore/tables/data_lastfm.csv")

# COMMAND ----------

from pyspark.sql.functions import col, explode, sum, count, expr
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Transformer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import Param
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS, ALSModel
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading and preprocessing the data

# COMMAND ----------

lastfm_df = spark.read.format("csv").option("inferSchema", "true").option("header","true").\
load("dbfs:/FileStore/tables/data_lastfm.csv").\
withColumn('play_count', col('play_count').cast('integer'))

lastfm_df.cache()

# COMMAND ----------

class clean_data(Transformer):
  
    def _transform(self, df):
      clean_df = df.dropna()
      return clean_df
      

class preprocess_data(Transformer):
  
  @keyword_only
  def __init__(self,cutoff_items=5, cutoff_users=750):
      super(preprocess_data, self).__init__()
      self.cutoff_items = cutoff_items
      self.cutoff_users = cutoff_users

  def _transform(self, df):
    df1 = df \
            .groupBy('user_sha1') \
            .agg(count('artist_name').alias('artist_name_count')) \
            .where(f'artist_name_count >= {self.cutoff_items}') \
            .select('user_sha1')

    df1 = df.join(df1, 'user_sha1', 'inner')

    df2 = df1 \
            .groupBy('artist_name') \
            .agg(count('user_sha1').alias('user_count')) \
            .where(f'user_count >= {self.cutoff_users}') \
            .select('artist_name')

    preprocessed_data = df.join(df2, 'artist_name', 'inner')

    return preprocessed_data


# COMMAND ----------

clean_data = clean_data()
lastfm_df = clean_data.transform(lastfm_df)

preprocess_data = preprocess_data()
lastfm_df = preprocess_data.transform(lastfm_df)

# COMMAND ----------

display(lastfm_df.take(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring the data

# COMMAND ----------

def get_sparsity(df, col_users, col_items, rating):
  num_users = df.select(col_users).distinct().count()
  num_artists = df.select(col_items).distinct().count()
  num_ratings = df.select(rating).count()
  sparsity = round((1 - (num_ratings)/(num_users*num_artists))*100,3)
  return sparsity

# COMMAND ----------

sparsity_rate = get_sparsity(lastfm_df, 'user_sha1', 'artist_name', 'play_count')
print(f'the data sparsity rate is:{sparsity_rate}')

# COMMAND ----------

# top 5 most played artists
display(lastfm_df.groupby('artist_name').agg(sum('play_count').alias("sum_play_count"))\
        .sort("sum_play_count", ascending=False).take(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training ALS

# COMMAND ----------

#converting user_sha1 and artist_name to integer indexes to fit als requirements

indexer_user_id = StringIndexer(inputCol="user_sha1", outputCol="user_id",handleInvalid="skip")
lastfm_indexed = indexer_user_id.fit(lastfm_df).transform(lastfm_df).\
withColumn('user_id', col('user_id').cast('integer'))

indexer_artist_id = StringIndexer(inputCol="artist_name", outputCol="artist_id",handleInvalid="skip")
index_artist_id_model = indexer_artist_id.fit(lastfm_indexed) 
lastfm_indexed = index_artist_id_model.transform(lastfm_indexed).\
withColumn('artist_id', col('artist_id').cast('integer'))

lastfm_indexed = lastfm_indexed.select('user_id', 'artist_id', 'play_count')

display(lastfm_indexed.head(5))

# COMMAND ----------

# model hyperparameters fine-tuning using cross-validation
als = ALS(userCol="user_id",
          itemCol="artist_id",
          ratingCol="play_count",
          nonnegative = True,
          implicitPrefs = True,
          coldStartStrategy="drop",
          checkpointInterval=1)

rankingEvaluator = RankingEvaluator()

param_grid = ParamGridBuilder()\
            .addGrid(als.rank, [5,10,20])\
            .addGrid(als.alpha,[1,10,40,60])\
            .addGrid(als.regParam, [.01, .1, 1, 10,100]) \
            .addGrid(als.maxIter, [10, 12, 15])\
            .build()

cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid ,
                    evaluator=rankingEvaluator,
                    numFolds=5)

cvModel = cv.fit(lastfm_indexed)

# COMMAND ----------

(train, test) = lastfm_indexed.randomSplit([0.8, 0.2], seed = 1)

# COMMAND ----------

# model with the best hyper-paramters combination

als = ALS(userCol="user_id",
          itemCol="artist_id",
          ratingCol="play_count",
          nonnegative = False,
          implicitPrefs = True,
          coldStartStrategy="drop",
         alpha=40)
model = als.fit(train)

# COMMAND ----------

#saving the model
def save_model(model, path):
  model.save(path)
  
def load_model(path):
  model = ALSModel.load(path)
  return model

save_model(model, 'dbfs:/FileStore/als_model')

# COMMAND ----------

#utility function for evaluating the model

class RankingEvaluator(Evaluator):

    @keyword_only
    def __init__(self, k=10):
        super(RankingEvaluator, self).__init__()
        self.k = k

    def _evaluate(self, predictedDF):
      
        windowSpec = Window.partitionBy('user_id').orderBy(col('prediction').desc())
        perUserPredictedItemsDF = predictedDF \
            .select('user_id', 'artist_id', 'prediction', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(self.k)) \
            .groupBy('user_id') \
            .agg(expr('collect_list(artist_id) as items'))

        windowSpec = Window.partitionBy('user_id').orderBy(col('play_count').desc())
        perUserActualItemsDF = predictedDF \
            .select('user_id', 'artist_id', 'play_count', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(self.k)) \
            .groupBy('user_id') \
            .agg(expr('collect_list(artist_id) as items'))

        perUserItemsRDD = perUserPredictedItemsDF.join(F.broadcast(perUserActualItemsDF), 'user_id', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        if perUserItemsRDD.isEmpty():
            return 0.0

        rankingMetrics = RankingMetrics(perUserItemsRDD)
        metric = rankingMetrics.meanAveragePrecisionAt(self.k)
        return metric

# COMMAND ----------

#evaluating the model on the test set 

predictions_df = model.transform(test)
re = RankingEvaluator()
re.evaluate(predictions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving a dataframe with user_id, actual artists listened to, and recommended artists

# COMMAND ----------

# generating a recommendation of 10 artists for each user
rec = model.recommendForAllUsers(10)\
.selectExpr("user_id", "explode(recommendations) as recom")\
.select("user_id", "recom.artist_id")

# COMMAND ----------

# converting artist_id back to artist_name and grouping them in a list
labelReverse = IndexToString(inputCol="artist_id", outputCol='artist_name', labels = index_artist_id_model.labels)
rec_transformed = labelReverse.transform(rec)
final_rec = rec_transformed.groupBy('user_id').\
agg(expr("collect_list(artist_name) as artists"))

# COMMAND ----------

# joining the dataframe with artists recommended and artists actually listned to on user_id
predvsactual = labelReverse.transform(train).\
orderBy(col("user_id"), expr("play_count DESC")).groupBy('user_id').\
agg(expr("collect_list(artist_name) as artists_actual")).join(final_rec, ["user_id"]).rdd\
.map(lambda row: (row[0],row[1][:10],row[2]))

# COMMAND ----------

# exporting the previous dataframe via pickle
predvsactual_df = predvsactual.toDF().toPandas().rename(columns={"_1": "user_id", "_2": "Listened to", "_3":"Recommendations"})
predvsactual_df.to_pickle('recommendations-artists.pkl',protocol=4)

# COMMAND ----------

predvsactual_df.head()
