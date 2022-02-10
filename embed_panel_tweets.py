import findspark
findspark.init()
from transformers import BertTokenizer
from transformers import BertModel
import torch
from pyspark.sql import SparkSession, Window
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

import ujson as json
import time
import pandas as pd

num_iters_to_hold_cluster = 4

findspark.os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"
findspark.os.environ["PYSPARK_PYTHON"]=   "./environment/bin/python"
dates = ['2020-08-01','2020-08-16','2020-09-01','2020-09-16','2020-10-01','2020-10-16','2020-11-01','2020-11-16','2020-12-01','2020-12-16']

def run_spark_embedding(dt, user_bucket_start, num_iters=num_iters_to_hold_cluster):
    with SparkSession.builder.appName(f"t_embed_{dt.replace('-','')}")\
                            .config("spark.archives", "onnx_conda_env.tar.gz#environment")\
                            .config("spark.sql.execution.arrow.enabled", "true")\
                            .getOrCreate() as spark:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
        sample_uids = spark.read.parquet("/user/etorf/panel/panel_sample_final_uids")
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

        for i in range(user_bucket_start, user_bucket_start + num_iters):
            DATA_PATH = f"/user/nir/panel_tweets/created_at_bucket={dt}/user_id_bucket={str(i)}"
            schm = T.StructType.fromJson(json.load(open("/home/kjoseph/twitterUSVoters/spark_analysis/tweet_schema.json")))
            schm.add("created_at_bucket", T.StringType(), False, None)
            schm.add("user_id_bucket", T.IntegerType(), False, None)
            panel_tweets = spark.read.json(DATA_PATH, schema=schm)
            panel_tweets = panel_tweets.select("id", panel_tweets.user.id.alias("user_id"), "full_text", "user_id_bucket")
            sample_tweets = panel_tweets.join(sample_uids, on=F.col("panel_uid") == F.col("user_id")).drop("panel_uid")

            t0 = time.time()

            print(f"started embedding DATA_PATH={DATA_PATH}")

            embed_df = sample_tweets.repartition(60).withColumn("embed", embed_batch_udf(F.col('full_text')))
            embed_df.write.save(
                path="/user/etorf/panel_sample_embeddings", format="parquet", mode="append", compression="gzip", partitionBy=["user_id_bucket"])

            print(f"it took {time.time() - t0} seconds")

for dt in dates:
    for u_bucket in range(0, 40, num_iters_to_hold_cluster):
        run_spark_embedding(dt, u_bucket)



