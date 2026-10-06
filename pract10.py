import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack

# ---------- Load dataset (CSV must be in the same folder as this script) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fake_news.csv")

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# ---------- Clean ----------
df = df.dropna(subset=["title", "text", "label"])
df["label"] = df["label"].astype(str).str.strip().str.lower().map({"real": 0, "fake": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

df["source"] = df["source"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")

# Title + text combined
df["full_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

print("\nLabel counts:")
print(df["label"].value_counts())

# ---------- Features ----------
# 1) Text features (TF-IDF)
text_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
text_features = text_vectorizer.fit_transform(df["full_text"])

# 2) Metadata features (replaces image features): source + category
encoder = OneHotEncoder(handle_unknown="ignore")
meta_features = encoder.fit_transform(df[["source", "category"]])

# 3) Simple numeric text features
df["word_count"] = df["text"].astype(str).str.split().str.len()
df["title_len"] = df["title"].astype(str).str.len()
num = df[["word_count", "title_len"]].values.astype(float)
num = (num - num.mean(axis=0)) / (num.std(axis=0) + 1e-9)

X = hstack([text_features, meta_features, num]).tocsr()
y = df["label"]

# ---------- Train / test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------- Results ----------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
print("Model training and evaluation completed.")