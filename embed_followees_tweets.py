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
import logging

findspark.os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"
findspark.os.environ["PYSPARK_PYTHON"]=   "./environment/bin/python"
start_date = datetime.datetime(2020, 8, 1)
end_date = datetime.datetime(2020, 11, 30)

def run_spark_embedding(start_dt, num_days=1):
    with SparkSession.builder.appName(f"t_embed_{start_dt.replace('-','')}")\
                            .config("spark.archives", "onnx_conda_env.tar.gz#environment")\
                            .config("spark.sql.execution.arrow.enabled", "true")\
                            .getOrCreate() as spark:
        logging.info(f"Spark started")
        for day in range(num_days):
            dt = start_dt.strftime("%Y-%m-%d") # convert to string
            for i in [(0,19), (20,39)]: # further split user_bucket to two to shorten each cycle
                spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
                followee_uids = spark.read.parquet("/user/etorf/panel/panel_sample_final_followee_uids")
                followee_uids = followee_uids.withColumn('user_id_bucket', followee_uids.followee_uid % 40)
                followee_uids = followee_uids.withColumn("user_id_bucket", followee_uids.user_id_bucket.cast('string'))
                followee_uids = followee_uids.filter(f"user_id_bucket >= {str(i[0])} AND user_id_bucket <= {str(i[1])}")

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

                logging.info(f"Started embedding for dt={start_dt} and user_id_bucket in {str(i)}")

                embed_df = sample_tweets.drop_duplicates().repartition(120, "user_id").withColumn("embed", embed_batch_udf(F.col('text'))).drop("text")
                embed_df.write.save(
                    path="/user/etorf/followee_sample_embeddings", format="parquet", mode="append", compression="gzip")

                logging.info(f"Finished embedding. It took {time.time() - t0} seconds")

                start_dt += datetime.timedelta(days=1) #add 1 day to continue to next day

        spark.stop()

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', filename="log_embed_spark.log", filemode="a")
logging.info("*************************Starting**********************************")

cur_date = start_date
while cur_date <= end_date:
    run_spark_embedding(cur_date, cur_date + datetime.timedelta(days=5))
    cur_date += datetime.timedelta(days=6)