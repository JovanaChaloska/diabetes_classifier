"""Reusable data-loading and feature-transformation methods.


All transformations are pure functions (load_dataset, prepare_label, clean_dataset) 
plus a Spark ML class Pipeline builder (build_preprocessing_pipeline) 
that produces a single features vector column. 
The classifier is then appended to the same pipeline so that fitting /
scoring is a single PipelineModel; this model can be saved to disk and
later loaded + transform-ed on the online data without any code duplication.
"""
from __future__ import annotations

from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from jobs.config import (
    ALL_FEATURES,
    BINARY_COLS,
    CONTINUOUS_COLS,
    LABEL_COL,
    LABEL_OUTPUT,
    ORDINAL_COLS,
)


def cast_known_columns(df: DataFrame) -> DataFrame:
    for col_name in [LABEL_COL] + ALL_FEATURES:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    return df


def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.csv(path, header=True, inferSchema=True)
    return cast_known_columns(df)


def prepare_label(df: DataFrame) -> DataFrame:
    return df.withColumn(LABEL_OUTPUT, F.col(LABEL_COL).cast(DoubleType()))


def clean_dataset(df: DataFrame) -> DataFrame:
    existing = [c for c in [LABEL_COL] + ALL_FEATURES if c in df.columns]
    return df.na.drop(subset=existing)


def prepare_dataset(spark: SparkSession, path: str) -> DataFrame:
    df = load_dataset(spark, path)
    df = prepare_label(df)
    df = clean_dataset(df)
    return df


def prepare_features(df: DataFrame) -> DataFrame:
    df = cast_known_columns(df)
    df = clean_dataset(df)
    return df


def build_feature_stages() -> List:
    continuous_assembler = VectorAssembler(
        inputCols=CONTINUOUS_COLS,
        outputCol="continuous_raw",
        handleInvalid="keep",
    )
    continuous_scaler = StandardScaler(
        inputCol="continuous_raw",
        outputCol="continuous_scaled",
        withMean=True,
        withStd=True,
    )
    final_assembler = VectorAssembler(
        inputCols=BINARY_COLS + ORDINAL_COLS + ["continuous_scaled"],
        outputCol="features",
        handleInvalid="keep",
    )
    return [continuous_assembler, continuous_scaler, final_assembler]


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(stages=build_feature_stages())
