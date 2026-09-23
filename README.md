# Diabetes Classifier (Spark, offline + online)

Two-stage diabetes classifier (`Diabetes_012`: 0 = no diabetes,
1 = prediabetes, 2 = diabetes) from the BRFSS 2015 Health Indicators
dataset.

- **Offline phase** — trains three Spark ML classifiers with
  cross-validation, picks the best by F1, saves the winning
  `PipelineModel` to `models/best_model/`.
- **Online phase** — a Python producer streams
  `data/offline.csv` row by row (without the label) into Kafka topic
  `health_data`. A Spark Structured Streaming job reads from that topic,
  applies the exact same preprocessing baked into the saved
  `PipelineModel`, enriches each record with the predicted class, and
  writes it to topic `health_data_predicted`.

Both phases run on an Apache Spark cluster spun up locally via Docker
Compose using the official public `apache/spark:3.5.3` image. Kafka +
Zookeeper come from a separate compose file ported from
`previous_work/docker-compose.yml`, attached to the same external Docker
network so the streaming job can reach the broker at `kafka:29092`.

## Table of contents

- [Layout](#layout)
- [Quick start — offline training](#quick-start--offline-training)
- [Quick start — online / streaming](#quick-start--online--streaming)
- [Classifiers & hyperparameter grids](#classifiers--hyperparameter-grids)
- [Model output layout](#model-output-layout)
- [Environment variables](#environment-variables)

## Layout

```
diabeter_classifier/
├── data/                   # offline.csv, online.csv (gitignored)
│   └── split_dataset.py    # Regenerates the 80/20 stratified split
├── docker/
│   ├── Dockerfile          # Extends apache/spark:3.5.3 with numpy
│   └── Dockerfile.producer # python:3.11-slim + kafka-python for the producer
├── jobs/
│   ├── config.py           # Paths, schema, Kafka + hyperparameter knobs
│   ├── preprocessing.py    # Reusable load / clean / feature pipeline
│   ├── train_offline.py    # Offline training entrypoint
│   ├── produce_offline.py  # Kafka producer (runs in diabetes-producer container)
│   └── predict_stream.py   # Spark Structured Streaming predictor
├── models/                 # Serialized best model (created at runtime)
├── docker-compose.yml      # Spark master + 2 workers
├── docker-compose.kafka.yml # Kafka + Zookeeper (ported from previous_work)
├── requirements.txt
└── README.md
```

## Quick start — offline training

Requirements: Docker Desktop (or any Docker engine shipping `docker compose`
v2) and ~2 GB of free disk for the Spark image.

Run from the repo root, in this order:

```powershell
# 0. Create the shared external docker network (one time per machine).
#    The Spark stack and the Kafka stack both attach to it.
docker network create diabetes-net

# 1. Build the local Spark image (apache/spark:3.5.3 + numpy).
#    Only needed the first time, or after editing docker/Dockerfile.
docker compose build

# 2. Start the cluster: 1 master + 2 workers, each with 2 cores / 3 GB RAM.
docker compose up -d

# 3. Submit the training job to the cluster. This reads data/offline.csv,
#    fits all three classifiers with cross-validation, evaluates them on a
#    held-out split, and writes the best one to ./models/best_model.
docker compose exec spark-master spark-submit `
    --master spark://spark-master:7077 `
    --deploy-mode client `
    --conf spark.driver.memory=2g `
    --conf spark.executor.memory=2g `
    /opt/app/jobs/train_offline.py

# 4. (Optional) Watch logs from another terminal while training runs.
docker compose logs -f spark-master

# 5. Shut the cluster down when you're done.
docker compose down
```

What each step does:

0. **`docker network create diabetes-net`** — creates the shared external
   network. Both `docker-compose.yml` and `docker-compose.kafka.yml` declare
   it as `external: true` so they can put their containers on it without
   owning it. Only needed once per machine; subsequent `up -d` commands
   reuse it.
1. **`docker compose build`** — builds the `diabetes-spark:3.5.3` image from
   [`docker/Dockerfile`](docker/Dockerfile). The base `apache/spark:3.5.3`
   image ships Python 3.8 but no `numpy`, so we add it here (pinned to
   `1.24.4`, the last version that supports Python 3.8). `pyspark.ml` imports
   numpy on load, so the job won't start without this step.
2. **`docker compose up -d`** — starts three containers defined in
   [`docker-compose.yml`](docker-compose.yml):
   - `spark-master` — runs `org.apache.spark.deploy.master.Master` on
     port 7077 (RPC) and exposes the master web UI on
     [http://localhost:8080](http://localhost:8080).
   - `spark-worker-1` / `spark-worker-2` — each runs
     `org.apache.spark.deploy.worker.Worker`, register with the master, and
     contribute 2 cores / 3 GB RAM (total cluster: 4 cores, 6 GB).
     Worker UIs: [localhost:8081](http://localhost:8081) and
     [localhost:8082](http://localhost:8082).
   - The whole repo is bind-mounted at `/opt/app` in every container, so
     data, code, and the output `models/` directory are shared between host
     and cluster.
3. **`spark-submit …`** — launches the driver inside the master container
   (`--deploy-mode client`). The driver connects to the cluster, distributes
   work to the two workers, and exposes its own application UI at
   [http://localhost:4040](http://localhost:4040) while the job is running.
   Output is streamed to your terminal.
4. **`docker compose logs -f spark-master`** — optional live tail of the
   master container logs if you'd rather run step 3 in the background.
5. **`docker compose down`** — stops and removes the three containers. The
   `diabetes-spark:3.5.3` image and the `./models/` directory on the host
   stay intact.


## Quick start — online / streaming

Prerequisites: the offline phase has already written
`models/best_model/`, and the shared `diabetes-net` docker network exists
(see step 0 of the offline section). Everything below runs inside Docker
containers — no host-side Python setup required.

From the repo root:

```powershell
# 1. Build the producer image (python + kafka-python). Only needed once.
docker compose build producer

# 2. Start Kafka + Zookeeper on the shared network (separate compose file).
docker compose -f docker-compose.kafka.yml up -d

# 3. Pre-create both topics. The Spark-Kafka connector requires the input
#    topic to exist *before* the streaming query starts.
docker compose -f docker-compose.kafka.yml exec kafka `
    kafka-topics --bootstrap-server localhost:9092 `
                 --create --if-not-exists `
                 --topic health_data --partitions 1 --replication-factor 1
docker compose -f docker-compose.kafka.yml exec kafka `
    kafka-topics --bootstrap-server localhost:9092 `
                 --create --if-not-exists `
                 --topic health_data_predicted --partitions 1 --replication-factor 1

# 4. Make sure the Spark cluster is up (from the offline section).
docker compose up -d

# 5. Start the Structured-Streaming predictor on the Spark cluster.
#    --packages pulls the Spark-Kafka connector from Maven on first run;
#    spark.jars.ivy redirects the Ivy cache to /tmp because the spark
#    user's $HOME isn't writable in the base image.
#    Leave this running in its own terminal.
docker compose exec spark-master spark-submit `
    --master spark://spark-master:7077 `
    --deploy-mode client `
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 `
    --conf spark.jars.ivy=/tmp/.ivy2 `
    --conf spark.driver.memory=2g `
    --conf spark.executor.memory=2g `
    /opt/app/jobs/predict_stream.py

# 6. In another terminal, fire up the producer container. It reads
#    data/offline.csv, drops the Diabetes_012 column, and publishes one
#    JSON record per row to topic 'health_data' on kafka:29092.
docker compose run --rm producer

# 7. Inspect the predictions flowing into 'health_data_predicted' (use the
#    broker's built-in console consumer — no local tools required).
docker compose -f docker-compose.kafka.yml exec kafka `
    kafka-console-consumer --bootstrap-server localhost:9092 `
                           --topic health_data_predicted `
                           --from-beginning `
                           --max-messages 5

# 8. Tear everything down when you're done.
docker compose down
docker compose -f docker-compose.kafka.yml down
```

What each step does:

1. **`docker compose build producer`** — builds the
   `diabetes-producer:latest` image from
   [`docker/Dockerfile.producer`](docker/Dockerfile.producer)
   (`python:3.11-slim` + `kafka-python==2.0.2`). The producer's Python
   code isn't baked in; it's bind-mounted at `/opt/app` on every `run`.
2. **`docker compose -f docker-compose.kafka.yml up -d`** — starts the
   Confluent 7.4.0 Zookeeper and Kafka containers (same images and
   listener config as
   [`previous_work/docker-compose.yml`](previous_work/docker-compose.yml)).
   The broker advertises two listeners: `PLAINTEXT://kafka:29092` for
   clients on `diabetes-net` (the Spark cluster and the producer
   container) and `EXTERNAL://localhost:9092` for any host-side tool.
3. **`kafka-topics --create …`** — pre-creates both Kafka topics. This is
   not optional: the Spark-Kafka source fails with
   `UnknownTopicOrPartitionException` if the subscribed topic doesn't
   exist when the query starts, even though the broker has
   `auto.create.topics.enable=true` for producers.
4. **`docker compose up -d`** — the Spark stack. If it's already running
   from the offline step, nothing happens; otherwise it comes up on
   `diabetes-net`.
5. **`predict_stream.py`** ([source](jobs/predict_stream.py)) — loads the
   fitted `PipelineModel` from `models/best_model/`, subscribes to the
   `health_data` topic at `kafka:29092`, parses each JSON payload into the
   21 feature columns, reuses `prepare_features` from
   [`jobs/preprocessing.py`](jobs/preprocessing.py) to cast + null-drop,
   then calls `model.transform` (which runs the preprocessing pipeline +
   classifier end-to-end). The predicted class is appended as a
   `prediction` field and the enriched record is republished as JSON to
   `health_data_predicted`. Checkpoints are kept at
   `/tmp/checkpoints/predict_stream/` inside the driver container
   (the default avoids Windows-bind-mount filesystem quirks in Spark's
   checkpoint manager; override via `STREAM_CHECKPOINT_DIR` if you want
   them on a named volume).
6. **`docker compose run --rm producer`** — spawns a one-shot
   `diabetes-producer` container on `diabetes-net` that runs
   [`jobs/produce_offline.py`](jobs/produce_offline.py). It reads
   `/opt/app/data/offline.csv` (bind-mounted from the host) one row at a
   time, strips `Diabetes_012`, serializes the remaining columns as JSON,
   and sends them to `health_data` on `kafka:29092` (the
   `KAFKA_BOOTSTRAP_HOST` env var is pre-set in the compose service).
   Default pacing is 50 ms between messages. Container-specific flag:
   `--rm` deletes the container when it exits so nothing lingers.
7. **`kafka-console-consumer …`** — the Confluent broker image ships a
   console consumer, so you can confirm predictions are landing on the
   output topic without installing anything on the host. Each message is a
   JSON object containing all 21 features, an `ingest_timestamp`, and the
   `prediction` (0 / 1 / 2).
8. **`docker compose down` (both files)** — stops and removes containers.
   The `diabetes-net` network itself persists; remove it manually with
   `docker network rm diabetes-net` if you're cleaning up.

### Streaming smoke test

Send just a few rows with no inter-record delay:

```powershell
docker compose run --rm producer --limit 20 --delay 0
```

Flags after the service name are forwarded to `jobs.produce_offline`
(the image uses it as `ENTRYPOINT`). The streaming job does not
auto-terminate (it's a long-running query) — press `Ctrl-C` on the
`spark-submit` terminal to stop it cleanly.


## Classifiers & hyperparameter grids

Three Spark ML classifier families, each with a small grid tuned by
`CrossValidator` using `numFolds=3` and an F1 evaluator:

| Family | Estimator | Grid (4 combinations) |
| ------ | --------- | --------------------- |
| Logistic regression | `LogisticRegression(family="multinomial", maxIter=50)` | `regParam ∈ {0.0, 0.1}` × `elasticNetParam ∈ {0.0, 0.5}` |
| Random forest | `RandomForestClassifier(seed=SEED)` | `numTrees ∈ {20, 50}` × `maxDepth ∈ {5, 10}` |
| Decision tree | `DecisionTreeClassifier(seed=SEED)` | `maxDepth ∈ {5, 10}` × `minInstancesPerNode ∈ {1, 10}` |

That's 12 fits per family × 3 families = 36 model fits per run on a
160 K-row training set. Tree-based models run distributed across both
workers; logistic regression is bounded by the driver's compute-per-fold.


## Environment variables

All defined in [`jobs/config.py`](jobs/config.py). Set them on the host
(local run) or via `docker compose exec -e NAME=value` (Docker run).

| Variable                   | Default                            | Description                                                |
| -------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| `APP_ROOT`                 | `/opt/app`                         | Base dir the other paths derive from.                      |
| `OFFLINE_DATA_PATH`        | `$APP_ROOT/data/offline.csv`       | Path to training / producer CSV.                           |
| `ONLINE_DATA_PATH`         | `$APP_ROOT/data/online.csv`        | Path to online CSV (reserved for future experiments).      |
| `MODEL_DIR`                | `$APP_ROOT/models`                 | Where `best_model/` is written.                            |
| `NUM_FOLDS`                | `3`                                | CV fold count.                                             |
| `TEST_FRACTION`            | `0.2`                              | Size of the held-out evaluation split.                     |
| `CV_PARALLELISM`           | `2`                                | Parallel grid-point evaluations per CV.                    |
| `KAFKA_BOOTSTRAP_HOST`     | `localhost:9092`                   | Bootstrap servers for the host-side producer.              |
| `KAFKA_BOOTSTRAP_INTERNAL` | `kafka:29092`                      | Bootstrap servers used by the streaming job inside Docker. |
| `INPUT_TOPIC`              | `health_data`                      | Topic the producer writes to / the streamer reads from.    |
| `OUTPUT_TOPIC`             | `health_data_predicted`            | Topic the streamer writes enriched records to.             |
| `PRODUCER_DELAY_SECONDS`   | `0.05`                             | Seconds to sleep between producer messages.                |
| `STREAM_CHECKPOINT_DIR`    | `/tmp/checkpoints/predict_stream` (in-container) | Structured-streaming checkpoint location. |

Worker CPU/RAM comes from the `--cores` / `--memory` arguments in the
`spark-worker-*` `command` lists in
[`docker-compose.yml`](docker-compose.yml); change both and re-run
`docker compose up -d`.

