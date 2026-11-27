"""
Diabetes Risk Screening — machine learning classification
Oru Stanley Enowru — portfolio project (synthetic use of a public dataset)

Problem: diabetes can be carried for a long time without symptoms, and by the
time it shows it may be too late. This project screens patients likely to have
diabetes so medical intervention can happen faster.

Dataset: Pima Indians Diabetes dataset (public, UCI / Kaggle mirror via
jbrownlee/Datasets on GitHub) — 768 female patients of Pima Indian heritage,
aged 21+. Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI, DiabetesPedigreeFunction, Age. Target: Outcome (1 = diabetes).

Pipeline: source -> clean (biologically impossible zeros treated as missing,
median-imputed, with counts) -> train/validation/test split (60/20/20,
stratified, seed 42) -> 3 models -> evaluate (accuracy, precision, recall,
F1, ROC-AUC) -> pick best, favouring recall (for screening, a missed case
is more costly than a false alarm) and F1.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)

SEED = 42
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv("data/pima-indians-diabetes.csv", header=None, names=COLS)
n_raw = len(df)
print(f"Raw rows: {n_raw}")
print(f"Positive (diabetes) in raw data: {int(df['Outcome'].sum())} "
      f"({df['Outcome'].mean()*100:.1f}%)")

# --- Cleaning: 0 is biologically impossible in these columns -> missing ---
zero_counts = {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING}
print("Zeros treated as missing:", zero_counts)
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
missing_before = int(df[ZERO_AS_MISSING].isna().sum().sum())
print(f"Total missing values after zero-replacement: {missing_before}")
medians = df[ZERO_AS_MISSING].median()
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].fillna(medians)
print("Missing after median imputation:", int(df.isna().sum().sum()))
print("Imputation medians:", {c: round(float(medians[c]), 1) for c in ZERO_AS_MISSING})

X = df.drop(columns="Outcome")
y = df["Outcome"]

# --- Split: train / validation / test 60 / 20 / 20, stratified ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print(f"Split sizes — train {len(X_train)}, validation {len(X_val)}, "
      f"test {len(X_test)}")
print(f"Positive rate — train {y_train.mean():.3f}, val {y_val.mean():.3f}, "
      f"test {y_test.mean():.3f}")

# --- Models ---
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}

def evaluate(model, X_, y_):
    pred = model.predict(X_)
    proba = model.predict_proba(X_)[:, 1]
    return {
        "accuracy": accuracy_score(y_, pred),
        "precision": precision_score(y_, pred, zero_division=0),
        "recall": recall_score(y_, pred, zero_division=0),
        "f1": f1_score(y_, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_, proba),
    }

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = {
        "validation": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }
    v, t = results[name]["validation"], results[name]["test"]
    print(f"\n{name}\n  val : " + ", ".join(f"{k}={v[k]:.3f}" for k in v))
    print(f"  test: " + ", ".join(f"{k}={t[k]:.3f}" for k in t))

# Best model: highest test F1, with recall as the screening tie-breaker.
best_name = max(results, key=lambda n: (results[n]["test"]["f1"],
                                        results[n]["test"]["recall"]))
print(f"\nBest model (test F1, recall tie-break): {best_name}")
best = models[best_name]

# --- Charts: ROC curves (all models) + confusion matrix (best, test set) ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC Curves — Diabetes Screening (test set)")
fig.tight_layout()
fig.savefig("charts/roc_curves.png", dpi=150)
plt.close(fig)

cm = confusion_matrix(y_test, best.predict(X_test))
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot(ax=ax)
ax.set_title(f"Confusion Matrix — {best_name} (test set)")
fig.tight_layout()
fig.savefig("charts/confusion_matrix.png", dpi=150)
plt.close(fig)
print("Confusion matrix (best, test): TN, FP, FN, TP =", cm.ravel().tolist())
print("Charts saved: charts/roc_curves.png, charts/confusion_matrix.png")

# --- Summary file ---
with open("results_summary.txt", "w") as f:
    f.write("DIABETES RISK SCREENING — RESULTS SUMMARY\n")
    f.write("Dataset: Pima Indians Diabetes (public, 768 rows)\n\n")
    f.write(f"Raw rows: {n_raw}; positive rate: {df['Outcome'].mean()*100:.1f}%\n")
    f.write(f"Zeros treated as missing: {zero_counts}\n")
    f.write(f"Total missing values imputed (median): {missing_before}\n")
    f.write(f"Split: train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}\n\n")
    for name, r in results.items():
        f.write(f"{name}\n")
        for split in ("validation", "test"):
            m = r[split]
            f.write(f"  {split}: accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
                    f"recall={m['recall']:.3f} f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f}\n")
    f.write(f"\nBest model: {best_name}\n")
    f.write(f"Confusion matrix (best, test) TN/FP/FN/TP: {cm.ravel().tolist()}\n")
print("Summary saved: results_summary.txt")
