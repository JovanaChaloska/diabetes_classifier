import os

# Paths (inside the container; the repo root is mounted at /opt/app).
# Override via environment variables if running outside docker.
APP_ROOT = os.environ.get("APP_ROOT", "/opt/app")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(APP_ROOT, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(APP_ROOT, "models"))

OFFLINE_DATA_PATH = os.environ.get(
    "OFFLINE_DATA_PATH", os.path.join(DATA_DIR, "offline.csv")
)
ONLINE_DATA_PATH = os.environ.get(
    "ONLINE_DATA_PATH", os.path.join(DATA_DIR, "online.csv")
)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model.info.txt")

LABEL_COL = "Diabetes_012"
LABEL_OUTPUT = "label"

# Binary 0/1 indicator features.
BINARY_COLS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

# Ordinal / small-integer features (already encoded numerically).
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]

# Continuous numeric features (will be standardized).
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

ALL_FEATURES = BINARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS

# Training hyperparameters
SEED = 42
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 3))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", 0.2))
CV_PARALLELISM = int(os.environ.get("CV_PARALLELISM", 2))

# Kafka / streaming
# From the host (producer script): use the EXTERNAL listener.
KAFKA_BOOTSTRAP_HOST = os.environ.get("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
# From a Spark container (streaming job): use the internal listener that the
# broker advertises as PLAINTEXT://kafka:29092.
KAFKA_BOOTSTRAP_INTERNAL = os.environ.get(
    "KAFKA_BOOTSTRAP_INTERNAL", "kafka:29092"
)
INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "health_data")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "health_data_predicted")

# Сeconds to wait between successive producer messages
PRODUCER_DELAY_SECONDS = float(os.environ.get("PRODUCER_DELAY_SECONDS", 0.05))

# Streaming checkpoint directory. Kept inside the container (not on the
# Windows bind mount at APP_ROOT) because the FileContext-based checkpoint
# manager Spark uses on structured streaming can fail to create atomic
# files on Windows-hosted bind mounts. 
STREAM_CHECKPOINT_DIR = os.environ.get(
    "STREAM_CHECKPOINT_DIR", "/tmp/checkpoints/predict_stream"
)
