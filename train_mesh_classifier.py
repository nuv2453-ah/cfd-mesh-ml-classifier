"""
Mesh Quality Classifier - training script
=========================================

Trains a Gaussian Naive Bayes classifier (and Random Forest + Logistic
Regression for comparison) on the CSV produced by simscale_log_parser.py.

Predicts mesh quality: "good" / "marginal" / "bad".

USAGE
-----
    pip install pandas scikit-learn matplotlib
    python train_mesh_classifier.py mesh_dataset.csv

REQUIRES at least ~15 labeled rows for results to mean anything;
30-50 is the sweet spot for a beginner project.
"""

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


# Columns that are NOT features (identifiers, raw outputs, the label itself)
NON_FEATURE_COLS = {
    "run_id",
    "label",
    "final_Cl",
    "final_Cd",
    "converged",
}


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise SystemExit(
            "ERROR: CSV has no 'label' column. Add labels in RUN_METADATA "
            "(good/marginal/bad) and re-generate the CSV."
        )
    df = df.dropna(subset=["label"])

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["label"]

    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")
    print(f"Class distribution:\n{y.value_counts().to_string()}\n")
    return X, y, feature_cols


def evaluate(name: str, model, X, y) -> None:
    """Cross-validate + print a held-out classification report."""
    if len(X) < 6:
        print(f"--- {name} ---")
        print("  Not enough data for meaningful evaluation (need >= 6 rows).\n")
        return

    cv = min(5, len(X))
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"--- {name} ---")
    print(f"  {cv}-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y if y.value_counts().min() >= 2 else None
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print("  Held-out classification report:")
    print(classification_report(y_te, y_pred, zero_division=0))
    print(f"  Confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_te, y_pred)}\n")


def main(csv_path: Path) -> None:
    X, y, feature_cols = load_dataset(csv_path)

    # Naive Bayes - the project headliner
    nb = Pipeline([("scale", StandardScaler()), ("nb", GaussianNB())])
    evaluate("Gaussian Naive Bayes", nb, X, y)

    # Comparison models
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    evaluate("Random Forest", rf, X, y)

    lr = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])
    evaluate("Logistic Regression", lr, X, y)

    # Feature importance from RF (interpretable bonus)
    if len(X) >= 6:
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=feature_cols)
        print("Top 10 most predictive features (Random Forest):")
        print(importances.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_mesh_classifier.py mesh_dataset.csv")
        sys.exit(1)
    main(Path(sys.argv[1]))




    