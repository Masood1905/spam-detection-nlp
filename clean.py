# clean.py  -  Step 2: text cleaning (lowercase, remove URLs/numbers/punctuation, stop words, stemming)
import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)  # only downloads the first time

STOP = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text) -> str:
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()                                          # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " ", text)                     # 2. strip URLs
    text = re.sub(r"\d+", " ", text)                                  # 3. strip numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # 4. strip punctuation
    tokens = [stemmer.stem(w) for w in text.split() if w not in STOP and len(w) > 2]  # 5. stop words + stemming
    return " ".join(tokens)


if __name__ == "__main__":  # only runs when you call `python clean.py`, not when another file imports clean_text
    BASE = Path(__file__).resolve().parent
    df = pd.read_csv(BASE / "artifacts" / "data_loaded.csv")
    df["clean"] = df["text"].apply(clean_text)
    print(df[["text", "clean"]].head())
    df.to_csv(BASE / "artifacts" / "data_cleaned.csv", index=False)
    print("Saved artifacts/data_cleaned.csv")
