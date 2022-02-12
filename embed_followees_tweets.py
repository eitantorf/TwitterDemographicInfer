import findspark
findspark.init()
from transformers import BertTokenizer
from transformers import BertModel
import torch
from pyspark.sql import SparkSession, Window
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import datetime
import ujson as json
import time
import pandas as pd

num_iters_to_hold_cluster = 4

findspark.os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"
findspark.os.environ["PYSPARK_PYTHON"]=   "./environment/bin/python"
start_date = datetime.datetime(2020, 8, 1)
end_date = datetime.datetime(2020, 11, 30)

def run_spark_embedding(dt):
    dt = dt.strftime("%Y-%m-%d")
    for i in [(0,19), (20,39)]:
        # I want to occupy the cluster each time for ~30 min so I split userbucket to two runs
        with SparkSession.builder.appName(f"t_embed_{dt.replace('-','')}")\
                                .config("spark.archives", "onnx_conda_env.tar.gz#environment")\
                                .config("spark.sql.execution.arrow.enabled", "true")\
                                .getOrCreate() as spark:
            spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
            followee_uids = spark.read.parquet("/user/etorf/panel/panel_sample_final_followee_uids")
            followee_uids = followee_uids.withColumn('user_id_bucket', followee_uids.followee_uid % 40)
            followee_uids = followee_uids.filter(f"user_id_bucket >= {str(i[0])} AND user_id_bucket <= {str(i[1])}")
            followee_uids = followee_uids.withColumn("user_id_bucket", followee_uids.user_id_bucket.cast('string'))

            tokenizer = BertTokenizer.from_pretrained('DeepPavlov/bert-base-cased-conversational')
            model = BertModel.from_pretrained('DeepPavlov/bert-base-cased-conversational', return_dict=True)
            bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
            bc_model = spark.sparkContext.broadcast(model)

            @pandas_udf(T.ArrayType(T.DoubleType()))
            def embed_batch_udf(texts: pd.Series) -> pd.Series:
                with torch.no_grad():
                    input = bc_tokenizer.value(texts.tolist(), return_tensors="pt", padding=True, truncation=True,
                                               max_length=512)
                    all_embeddings = bc_model.value(input['input_ids'])['pooler_output'].tolist()
                    return pd.Series(all_embeddings)

            schm = T.StructType.fromJson(json.load(open("/home/nir/spark_stuff/decahose_schema.json")))
            schm.add("ds", T.StringType(), False, None)
            decahose_tweets = spark.read.json("/user/nir/decahose_tweets", schema=schm)
            decahose_tweets = decahose_tweets.filter(f"ds == '{dt}'")
            decahose_tweets = decahose_tweets.select("id", decahose_tweets.user.id.alias("user_id"), "text")

            sample_tweets = decahose_tweets.join(followee_uids, on=F.col("followee_uid")==F.col("user_id")).drop("followee_uid")

            t0 = time.time()

            print(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Started embedding for date={dt} and user_id_bucket in {str(i)}")

            embed_df = sample_tweets.drop_duplicates().repartition(60).withColumn("embed", embed_batch_udf(F.col('text'))).drop("text")
            embed_df.write.save(
                path="/user/etorf/followee_sample_embeddings", format="parquet", mode="append", compression="gzip", partitionBy=["user_id_bucket"])

            print(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: It took {time.time() - t0} seconds")

cur_date = start_date
while cur_date <= end_date:
    run_spark_embedding(cur_date)
    cur_date += datetime.timedelta(days=1)