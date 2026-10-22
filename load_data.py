# load_data.py  -  Step 1: load the SMS Spam Collection and remove duplicates
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
(BASE / "artifacts").mkdir(exist_ok=True)

try:
    df = pd.read_csv(BASE / "data" / "spam.csv", encoding="utf-8")
except UnicodeDecodeError:  # the original Kaggle download is Latin-1 encoded
    df = pd.read_csv(BASE / "data" / "spam.csv", encoding="latin-1")
if "v1" in df.columns:  # the original Kaggle file uses v1 (label) and v2 (text)
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

print("Loaded:", df.shape)
df = df.dropna(subset=["text"])

# The dataset contains hundreds of repeated messages. Keeping them would let the same
# message appear in both the training and the test set, which makes the scores look better than they are.
before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate messages -> {len(df)} unique messages")
print("Label counts:", df["label"].value_counts().to_dict())

df.to_csv(BASE / "artifacts" / "data_loaded.csv", index=False)
print("Saved artifacts/data_loaded.csv")
