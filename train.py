# train.py  -  Step 3: split the data, then fit TF-IDF + Naive Bayes on the TRAINING set only
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
(BASE / "models").mkdir(exist_ok=True)

df = pd.read_csv(BASE / "artifacts" / "data_cleaned.csv")
df["clean"] = df["clean"].fillna("")

# 80/20 split; stratify keeps the same spam/ham ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# The vectorizer is inside the pipeline, so its vocabulary and word weights are learned
# from the training messages only. (Fitting it on all messages would leak test data into training.)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
    ("model", MultinomialNB()),
])

# Choose the Naive Bayes smoothing value (alpha) with 5-fold cross-validation on the training set
search = GridSearchCV(
    pipeline,
    {"model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0]},
    scoring="f1_macro",
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
)
search.fit(X_train, y_train)
print("Best alpha (cross-validation):", search.best_params_["model__alpha"])
print("Trained on", len(X_train), "messages | test set:", len(X_test), "messages")

joblib.dump(search.best_estimator_, BASE / "models" / "spam_pipeline.pkl")
pd.DataFrame({"clean": X_train, "label": y_train}).to_csv(BASE / "artifacts" / "train_set.csv", index=False)
pd.DataFrame({"clean": X_test, "label": y_test}).to_csv(BASE / "artifacts" / "test_set.csv", index=False)
print("Saved models/spam_pipeline.pkl")
