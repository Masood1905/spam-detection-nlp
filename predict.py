# predict.py  -  classify any message from the command line
# Usage:  python predict.py "Congratulations! You have won a free prize. Call now"
import sys
from pathlib import Path

import joblib

from clean import clean_text

pipeline = joblib.load(Path(__file__).resolve().parent / "models" / "spam_pipeline.pkl")


def predict(text: str):
    probabilities = pipeline.predict_proba([clean_text(text)])[0]
    best = probabilities.argmax()
    return pipeline.classes_[best], probabilities[best]


if __name__ == "__main__":
    message = " ".join(sys.argv[1:]) or input("Type a message: ")
    label, confidence = predict(message)
    print(f"{label.upper()}  (confidence {confidence:.1%})")
