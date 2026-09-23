"""Offline training job.

Reads offline.csv, applies the shared preprocessing pipeline, and trains
three different classifiers, each tuned with CrossValidator over a
small grid of hyperparameters. The model with the best F1 score is serialized
to models/best_model so it can later be reloaded and applied to online classification.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyspark.ml import Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.classification import (  # noqa: E402
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # noqa: E402
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from jobs.config import (  # noqa: E402
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    CV_PARALLELISM,
    LABEL_OUTPUT,
    MODEL_DIR,
    NUM_FOLDS,
    OFFLINE_DATA_PATH,
    SEED,
    TEST_FRACTION,
)
from jobs.preprocessing import build_feature_stages, prepare_dataset  # noqa: E402


@dataclass
class CandidateResult:
    name: str
    cv_avg_f1: float          # mean F1 across CV folds for the best grid point
    test_f1: float            # F1 on the held-out split (for reporting only)
    best_params: dict         # chosen hyperparameters
    best_pipeline: PipelineModel  # fitted pipeline (preprocessing + classifier)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DiabetesOfflineTraining")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )


def f1_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(
        labelCol=LABEL_OUTPUT, predictionCol="prediction", metricName="f1"
    )


def _cross_validator(pipeline: Pipeline, grid, evaluator) -> CrossValidator:
    return CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        parallelism=CV_PARALLELISM,
        seed=SEED,
    )


def _extract_best_params(cv_model: CrossValidatorModel) -> dict:
    best_idx = int(max(
        range(len(cv_model.avgMetrics)), key=lambda i: cv_model.avgMetrics[i]
    ))
    best_param_map = cv_model.getEstimatorParamMaps()[best_idx]
    return {p.name: v for p, v in best_param_map.items}


def build_candidate_cvs(feature_stages: List) -> List[Tuple[str, CrossValidator]]:
    evaluator = f1_evaluator()

    lr = LogisticRegression(
        labelCol=LABEL_OUTPUT,
        featuresCol="features",
        family="multinomial",
        maxIter=50,
    )
    lr_pipeline = Pipeline(stages=feature_stages + [lr])
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )

    rf = RandomForestClassifier(
        labelCol=LABEL_OUTPUT,
        featuresCol="features",
        seed=SEED,
    )
    rf_pipeline = Pipeline(stages=feature_stages + [rf])
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [20, 50])
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )

    dt = DecisionTreeClassifier(
        labelCol=LABEL_OUTPUT,
        featuresCol="features",
        seed=SEED,
    )
    dt_pipeline = Pipeline(stages=feature_stages + [dt])
    dt_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [5, 10])
        .addGrid(dt.minInstancesPerNode, [1, 10])
        .build()
    )

    return [
        ("LogisticRegression", _cross_validator(lr_pipeline, lr_grid, evaluator)),
        ("RandomForest", _cross_validator(rf_pipeline, rf_grid, evaluator)),
        ("DecisionTree", _cross_validator(dt_pipeline, dt_grid, evaluator)),
    ]


def train_all(train_df, test_df) -> List[CandidateResult]:
    feature_stages = build_feature_stages()
    candidates = build_candidate_cvs(feature_stages)
    evaluator = f1_evaluator()

    results: List[CandidateResult] = []
    for name, cv in candidates:
        print(f"\n=== Training {name} with {NUM_FOLDS}-fold CV ===", flush=True)
        start = time.time()
        cv_model = cv.fit(train_df)
        elapsed = time.time() - start

        best_cv_f1 = float(max(cv_model.avgMetrics))
        best_params = _extract_best_params(cv_model)
        test_predictions = cv_model.bestModel.transform(test_df)
        test_f1 = float(evaluator.evaluate(test_predictions))

        print(
            f"[{name}] done in {elapsed:.1f}s | "
            f"CV best F1 = {best_cv_f1:.4f} | test F1 = {test_f1:.4f} | "
            f"params = {best_params}",
            flush=True,
        )

        results.append(
            CandidateResult(
                name=name,
                cv_avg_f1=best_cv_f1,
                test_f1=test_f1,
                best_params=best_params,
                best_pipeline=cv_model.bestModel,
            )
        )
    return results


def save_best_model(results: List[CandidateResult]) -> CandidateResult:
    best = max(results, key=lambda r: r.cv_avg_f1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isdir(BEST_MODEL_PATH):
        shutil.rmtree(BEST_MODEL_PATH)

    best.best_pipeline.write().overwrite().save(BEST_MODEL_PATH)
    print(
        f"\n### Best model: {best.name} "
        f"(CV F1 = {best.cv_avg_f1:.4f}, test F1 = {best.test_f1:.4f}) "
        f"saved to {BEST_MODEL_PATH}",
        flush=True,
    )

    with open(BEST_MODEL_INFO_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"best_algorithm={best.name}\n")
        fh.write(f"best_cv_f1={best.cv_avg_f1}\n")
        fh.write(f"best_test_f1={best.test_f1}\n")
        fh.write(f"best_params={best.best_params}\n\n")
        fh.write("--- All candidates ---\n")
        for r in results:
            fh.write(
                f"{r.name}: cv_f1={r.cv_avg_f1:.4f}, "
                f"test_f1={r.test_f1:.4f}, params={r.best_params}\n"
            )
    return best


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading offline data from {OFFLINE_DATA_PATH}", flush=True)
    df = prepare_dataset(spark, OFFLINE_DATA_PATH)
    total_rows = df.count()
    print(f"Total rows after cleaning: {total_rows:,}", flush=True)

    train_df, test_df = df.randomSplit(
        [1 - TEST_FRACTION, TEST_FRACTION], seed=SEED
    )
    train_df = train_df.cache()
    test_df = test_df.cache()
    print(
        f"Train rows: {train_df.count():,} | Test rows: {test_df.count():,}",
        flush=True,
    )

    results = train_all(train_df, test_df)
    save_best_model(results)

    spark.stop()


if __name__ == "__main__":
    main()
