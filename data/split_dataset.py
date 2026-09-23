import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "diabetes_012_health_indicators_BRFSS2015.csv"
LABEL_COL = "Diabetes_012"
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df[LABEL_COL].value_counts().sort_index()}\n")

offline_df, online_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[LABEL_COL],
)

offline_df.to_csv("offline.csv", index=False)
online_df.to_csv("online.csv", index=False)

print(f"Offline split: {len(offline_df)} samples")
print(f"Class distribution:\n{offline_df[LABEL_COL].value_counts().sort_index()}\n")

print(f"Online split:  {len(online_df)} samples")
print(f"Class distribution:\n{online_df[LABEL_COL].value_counts().sort_index()}\n")
