"""Frozen Waypoint-6m embeddings -> four classifiers, fresh random stratified 10-fold CV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE / "waypoint_embeddings_389.csv"
OUTPUT = HERE / "waypoint6m_random10fold_metrics.tsv"

# The NPZ supplies only sample IDs and ASD/Healthy labels; folds are generated in this run.
NPZ_OPTIONS = [
    HERE / "waypoint_agora_corrected.npz",
    HERE.parent / "14_10" / "waypoint_agora_corrected.npz",
    HERE.parent / "14_10" / "waypoint_agora_checked.npz",
    HERE / "waypoint_agora_checked.npz",
]
npz_path = next((p for p in NPZ_OPTIONS if p.is_file()), None)
if npz_path is None:
    raise FileNotFoundError(
        "Missing label NPZ. Put waypoint_agora_corrected.npz next to this script, "
        "or leave waypoint_agora_checked.npz in ../14_10/."
    )

with np.load(npz_path, allow_pickle=False) as z:
    ids = z["sample_ids"].astype(str)
    y = z["labels"].astype(int)

if not EMBEDDINGS.is_file():
    raise FileNotFoundError(f"Missing embedding CSV: {EMBEDDINGS}")
df = pd.read_csv(EMBEDDINGS)
columns = [f"dim_{i}" for i in range(256)]
if "Sample ID" not in df or any(c not in df for c in columns):
    raise ValueError("Expected Sample ID and dim_0..dim_255 in embedding CSV")
if len(df) != len(ids) or df["Sample ID"].duplicated().any():
    raise ValueError("Incorrect embedding count or duplicate sample identifiers")
if len(set(ids)) != len(ids) or set(df["Sample ID"]) != set(ids):
    raise ValueError("Embedding IDs do not match the NPZ sample IDs exactly")
X = df.set_index("Sample ID").loc[ids, columns].to_numpy(dtype=float)
if not np.isfinite(X).all():
    raise ValueError("Embedding matrix contains NaN or infinity")
if len(X) != 389 or X.shape[1] != 256 or np.bincount(y, minlength=2).tolist() != [181, 208]:
    raise ValueError("Unexpected cohort size, dimensions or ASD/Healthy counts")
splits = list(StratifiedKFold(
    n_splits=10, shuffle=True, random_state=42
).split(X, y))

# Same model settings as uploaded 01_baseline_ml.py; only CPU worker count is capped at 10.
models = {
    "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=42, n_jobs=10, verbosity=0),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=10),
}
metric_names = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1"]
rows = []
print(f"Inputs: {len(ids)} runs × {X.shape[1]} Waypoint features; ASD={sum(y==1)}, Healthy={sum(y==0)}")
print(f"Labels from: {npz_path} | fresh random stratified 10-fold CV, seed 42")
for name, estimator in models.items():
    per_fold = []
    print(f"\n{name}", flush=True)
    for k, (train, test) in enumerate(splits, start=1):
        fitted = clone(estimator).fit(X[train], y[train])
        pred = fitted.predict(X[test])
        score = (fitted.predict_proba(X[test])[:, 1]
                 if hasattr(fitted, "predict_proba")
                 else fitted.decision_function(X[test]))
        tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
        per_fold.append({
            "Accuracy": accuracy_score(y[test], pred),
            "AUC": roc_auc_score(y[test], score),
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "Precision": precision_score(y[test], pred, zero_division=0),
            "F1": f1_score(y[test], pred, zero_division=0),
        })
        print(f"  Fold {k:2d}/10: accuracy={per_fold[-1]['Accuracy']:.4f}, AUC={per_fold[-1]['AUC']:.4f}", flush=True)
    row = {"Method": "Frozen Waypoint-6m", "Model": name,
           "Features": "256.0 ± 0.0"}
    for metric in metric_names:
        values = np.array([r[metric] for r in per_fold])
        row[metric] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    rows.append(row)

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, sep="\t", index=False)
print("\n10-FOLD RESULTS (mean ± sample SD):\n" + result.to_string(index=False))
print(f"\nSaved ONE output: {OUTPUT}")
