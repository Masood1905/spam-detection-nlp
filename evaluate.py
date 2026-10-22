# evaluate.py  -  Step 4: test-set results, confusion matrix and a comparison with other models
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

BASE = Path(__file__).resolve().parent
(BASE / "images").mkdir(exist_ok=True)

pipeline = joblib.load(BASE / "models" / "spam_pipeline.pkl")
train = pd.read_csv(BASE / "artifacts" / "train_set.csv").fillna("")
test = pd.read_csv(BASE / "artifacts" / "test_set.csv").fillna("")

y_pred = pipeline.predict(test["clean"])
print("=" * 55)
print("FINAL MODEL: TF-IDF + Multinomial Naive Bayes "
      f"(alpha={pipeline.named_steps['model'].alpha})")
print("ACCURACY:", round(accuracy_score(test["label"], y_pred), 4))
print("\nCONFUSION MATRIX (rows = actual ham/spam, columns = predicted):")
print(confusion_matrix(test["label"], y_pred, labels=["ham", "spam"]))
print("\nCLASSIFICATION REPORT:")
print(classification_report(test["label"], y_pred, digits=3))

ConfusionMatrixDisplay.from_predictions(test["label"], y_pred, labels=["ham", "spam"], cmap="Blues")
plt.title("Spam detector: confusion matrix (test set)")
plt.tight_layout()
plt.savefig(BASE / "images" / "confusion_matrix.png", dpi=150)
print("Saved images/confusion_matrix.png")

# Compare with other models on exactly the same train/test split and features
candidates = {
    "Naive Bayes (alpha=1.0, untuned)": MultinomialNB(alpha=1.0),
    f"Naive Bayes (alpha={pipeline.named_steps['model'].alpha}, tuned)": MultinomialNB(
        alpha=pipeline.named_steps["model"].alpha),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Linear SVM (balanced)": LinearSVC(class_weight="balanced"),
}
rows = []
for name, clf in candidates.items():
    p = clone(pipeline).set_params(model=clf).fit(train["clean"], train["label"])
    pr = p.predict(test["clean"])
    cm = confusion_matrix(test["label"], pr, labels=["ham", "spam"])
    rows.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(test['label'], pr):.1%}",
        "Spam precision": f"{precision_score(test['label'], pr, pos_label='spam'):.1%}",
        "Spam recall": f"{recall_score(test['label'], pr, pos_label='spam'):.1%}",
        "Spam F1": f"{f1_score(test['label'], pr, pos_label='spam'):.3f}",
        "Legit flagged as spam": int(cm[0, 1]),
        "Spam missed": int(cm[1, 0]),
    })
print("\nMODEL COMPARISON (same test set):")
print(pd.DataFrame(rows).to_string(index=False))
