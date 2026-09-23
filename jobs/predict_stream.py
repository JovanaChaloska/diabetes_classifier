"""Spark Structured Streaming job: online prediction.

1. Load the PipelineModel trained offline from jobs.config.BEST_MODEL_PATH
2. Read the Kafka topic jobs.config.INPUT_TOPIC as a streaming DataFrame
3. Parse each JSON value into the feature schema, apply the same cast + null-drop transformations the offline job used
   (jobs.preprocessing.prepare_features), and hand the rows to the loaded PipelineModel. The pipelines preprocessing stages are applied
   automatically inside model.transform, no duplicated feature code
4. Each record gets the model's prediction (0 / 1 / 2) and is written back to Kafka topic jobs.config.OUTPUT_TOPIC as JSON
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

from jobs.config import (
    ALL_FEATURES,
    BEST_MODEL_PATH,
    INPUT_TOPIC,
    KAFKA_BOOTSTRAP_INTERNAL,
    OUTPUT_TOPIC,
    STREAM_CHECKPOINT_DIR,
)
from jobs.preprocessing import prepare_features


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DiabetesOnlinePrediction")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def build_input_schema() -> StructType:
    return StructType([StructField(c, DoubleType(), True) for c in ALL_FEATURES])


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading PipelineModel from {BEST_MODEL_PATH}", flush=True)
    model = PipelineModel.load(BEST_MODEL_PATH)

    print(
        f"Subscribing to Kafka topic {INPUT_TOPIC!r} "
        f"on {KAFKA_BOOTSTRAP_INTERNAL}",
        flush=True,
    )
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_INTERNAL)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Kafka columns: key, value (bytes), topic, partition, offset, timestamp
    schema = build_input_schema()
    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_value", "timestamp")
           .select(
               F.from_json(F.col("json_value"), schema).alias("data"),
               F.col("timestamp").alias("ingest_timestamp"),
           )
           .select("data.*", "ingest_timestamp")
    )

    # Reuse the offline preprocessing (cast + null-drop)
    features = prepare_features(parsed)

    scored = model.transform(features)

    # all original feature columns + predicted class (cast back to int) + ingest timestamp
    enriched = scored.select(
        *ALL_FEATURES,
        F.col("prediction").cast("int").alias("prediction"),
        F.col("ingest_timestamp"),
    )

    out = enriched.select(
        F.to_json(F.struct(*enriched.columns)).alias("value")
    )

    print(
        f"Writing predictions to topic {OUTPUT_TOPIC!r} "
        f"(checkpoint: {STREAM_CHECKPOINT_DIR})",
        flush=True,
    )
    query = (
        out.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_INTERNAL)
        .option("topic", OUTPUT_TOPIC)
        .option("checkpointLocation", STREAM_CHECKPOINT_DIR)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
