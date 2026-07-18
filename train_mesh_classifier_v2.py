import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

NON_FEATURE_COLS = {"run_id", "label"}

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["label"]
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")
    print(f"Class distribution:\n{y.value_counts().to_string()}\n")
    return X, y

def evaluate(name, model, X, y):
    cv = min(5, y.value_counts().min())
    if cv < 2:
        print(f"--- {name} ---\n  Smallest class has <2 members, skipping CV.\n")
        return
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"--- {name} ---")
    print(f"  {cv}-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

X, y = load_dataset("mesh_dataset_v2.csv")
nb = Pipeline([("scale", StandardScaler()), ("nb", GaussianNB())])
evaluate("Gaussian Naive Bayes", nb, X, y)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
evaluate("Random Forest", rf, X, y)
lr = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
evaluate("Logistic Regression", lr, X, y)
