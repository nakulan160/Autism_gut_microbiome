#!/usr/bin/env python3
"""Compare frozen Waypoint-45m and Waypoint-170m embeddings on the ASD 389-run cohort.

Run in the directory containing waypoint_389.parquet and, if desired,
waypoint_embeddings_389.csv (existing Waypoint-6m embeddings).

Uses random StratifiedKFold(10, shuffle=True, random_state=42), no tuning.
For labels, uses waypoint_agora_corrected.npz (or older checked NPZ) ONLY.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS = {
    "6m": ("outpost-bio/Waypoint-6m", 256, "waypoint_embeddings_389.csv"),
    "45m": ("outpost-bio/Waypoint-45m", 512, "waypoint45m_embeddings_389.csv"),
    "170m": ("outpost-bio/Waypoint-170m", 768, "waypoint170m_embeddings_389.csv"),
}
METRICS = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1"]
SEED = 42


def find_npz(directory: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Label NPZ missing: {p}")
        return p
    options = [
        directory / "waypoint_agora_corrected.npz",
        directory.parent / "14_10" / "waypoint_agora_corrected.npz",
        directory.parent / "14_10" / "waypoint_agora_checked.npz",
        directory / "waypoint_agora_checked.npz",
    ]
    found = next((p for p in options if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "Cannot find label NPZ. Use --npz /path/to/waypoint_agora_corrected.npz"
        )
    return found


def check_vocabulary(model_id: str, archive, name: str):
    """Compare the same 355 prepared groups against this model's vocabulary."""
    try:
        vocab_file = hf_hub_download(repo_id=model_id, filename="vocab.json")
    except Exception as exc:
        raise RuntimeError(
            f"No access to {model_id}. Request/accept access at "
            f"https://huggingface.co/{model_id} and authenticate on this GPU server. "
            f"Original error: {type(exc).__name__}: {str(exc)[:220]}"
        ) from exc

    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)
    paths = archive["waypoint_taxonomy_paths"].astype(str)
    mass = archive["waypoint_X_relative"].astype(float)
    token_names = [p.split(";")[-1].strip() for p in paths]
    recognised = np.asarray([t in vocab and int(vocab[t]) != 1 for t in token_names])
    denom = mass.sum(axis=1)
    if np.any(denom <= 0):
        raise ValueError("A sample has zero usable named mass")
    coverage = mass[:, recognised].sum(axis=1) / denom
    print(
        f"  Vocabulary {name}: {int(recognised.sum())}/{len(paths)} groups; "
        f"median named-abundance coverage={np.median(coverage):.2%}; "
        f"min={coverage.min():.2%}",
        flush=True,
    )
    if np.any(coverage <= 0):
        print(f"  WARNING: {(coverage <= 0).sum()} samples have zero recognized abundance", flush=True)
    return int(recognised.sum()), float(np.median(coverage))


def embeddings(directory: Path, tag: str, data: Path, batch_size: int) -> Path:
    model_id, width, output_name = MODELS[tag]
    output = directory / output_name
    if output.is_file():
        print(f"  Reusing existing {output.name} (dimensions checked next)", flush=True)
        return output
    if tag == "6m":
        raise FileNotFoundError(
            "Waypoint-6m reference CSV not found; rerun 6m embed or omit --include-6m"
        )
    exe = shutil.which("waypoint")
    if not exe:
        raise RuntimeError("Waypoint CLI missing. In the active environment: pip install waypoint-bio")
    cmd = [
        exe, "embed", "--model", model_id, "--data", str(data),
        "--output", str(output), "--pooling", "last_token",
        "--batch_size", str(batch_size), "--max_length", "512", "--device", "cuda",
    ]
    print("  Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=directory, check=True)
    return output


def load_embeddings(path: Path, ids: np.ndarray, width: int) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = [f"dim_{i}" for i in range(width)]
    if "Sample ID" not in frame.columns:
        raise ValueError(f"No Sample ID column in {path}")
    if len(frame) != len(ids) or frame["Sample ID"].duplicated().any():
        raise ValueError(f"Rows or duplicate sample IDs incorrect in {path}")
    if set(frame["Sample ID"].astype(str)) != set(ids):
        missing = set(ids).difference(frame["Sample ID"].astype(str))
        raise ValueError(f"IDs do not match NPZ in {path}: missing examples {list(missing)[:5]}")
    if any(c not in frame for c in columns):
        raise ValueError(f"Expected embedding dimension {width}; columns missing in {path}")
    X = frame.set_index("Sample ID").loc[ids, columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite embeddings in {path}")
    print(f"  Embeddings verified: {X.shape[0]} x {X.shape[1]}, unique IDs, all finite", flush=True)
    return X


def classifiers():
    # Same model parameters as the earlier 01_baseline_ml.py and 6m evaluation.
    return {
        "XGBoost": XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=10, verbosity=0,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=10),
    }


def evaluate(X: np.ndarray, y: np.ndarray, tag: str, n_vocab, median_cov):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    print(f"  Random stratified 10-fold CV; seed={SEED}; no hyperparameter tuning", flush=True)
    output = []
    for model_name, estimator in classifiers().items():
        print(f"    {model_name}:", end=" ", flush=True)
        scores = []
        for fold_id, (train, test) in enumerate(folds, 1):
            fitted = clone(estimator).fit(X[train], y[train])
            pred = fitted.predict(X[test])
            probability = (
                fitted.predict_proba(X[test])[:, 1] if hasattr(fitted, "predict_proba")
                else fitted.decision_function(X[test])
            )
            tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
            scores.append({
                "Accuracy": accuracy_score(y[test], pred),
                "AUC": roc_auc_score(y[test], probability),
                "Sensitivity": tp / (tp + fn),
                "Specificity": tn / (tn + fp),
                "Precision": precision_score(y[test], pred, zero_division=0),
                "F1": f1_score(y[test], pred, zero_division=0),
            })
            print(f"{fold_id}", end=" ", flush=True)
        row = {
            "Method": f"Frozen Waypoint-{tag}", "Model": model_name,
            "Features": f"{X.shape[1]:.1f} ± 0.0",
        }
        for m in METRICS:
            a = np.asarray([s[m] for s in scores])
            row[m] = f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"
        row["Vocab recognized / 355"] = f"{n_vocab}/355" if n_vocab is not None else ""
        row["Median named-abundance coverage"] = (
            f"{median_cov:.4f}" if median_cov is not None else ""
        )
        output.append(row)
        print(f" | Accuracy {row['Accuracy']} | AUC {row['AUC']}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd(), help="Data/output folder (default current directory)")
    parser.add_argument("--npz", help="Path to NPZ with IDs and labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-6m", action="store_true", help="Also compare existing 6m embeddings (does not download 6m)")
    parser.add_argument("--models", nargs="+", choices=["45m", "170m"], default=["45m", "170m"])
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    data = directory / "waypoint_389.parquet"
    if not data.exists():
        raise FileNotFoundError(f"Prepared Waypoint file missing: {data}")
    npz_file = find_npz(directory, args.npz)
    print(f"Using prepared input: {data}")
    print(f"Labels from: {npz_file}")
    with np.load(npz_file, allow_pickle=False) as z:
        ids = z["sample_ids"].astype(str)
        y = z["labels"].astype(int)
        assert ids.shape == (389,) and len(set(ids)) == 389, "Unexpected samples or duplicate IDs"
        assert np.bincount(y, minlength=2).tolist() == [181, 208], "Unexpected class counts"
        wanted = ["6m"] if args.include_6m else []
        wanted += args.models
        rows = []
        failed = []
        output = directory / "waypoint_model_sizes_random10fold_metrics.tsv"
        for tag in wanted:
            model_id, width, _ = MODELS[tag]
            print(f"\n=== {model_id}: expected embeddings 389 x {width} ===", flush=True)
            try:
                # For 6m, the vocabulary in this NPZ is precisely the one checked previously.
                if tag == "6m" and "waypoint_candidate_token_ids" in z:
                    n_vocab = int(np.count_nonzero(z["waypoint_candidate_token_ids"] != 1))
                    cov = float(np.median(z["waypoint_coverage_fraction_named"]))
                    print(f"  Stored 6m vocabulary: {n_vocab}/355 groups; median coverage={cov:.2%}")
                else:
                    n_vocab, cov = check_vocabulary(model_id, z, tag)
                csv_path = embeddings(directory, tag, data, args.batch_size)
                X = load_embeddings(csv_path, ids, width)
                rows.extend(evaluate(X, y, tag, n_vocab, cov))
            except Exception as err:
                failed.append(tag)
                print(f"\n  FAILED / NOT COMPLETED for {tag}: {err}", flush=True)
                if not rows:
                    print("  No classifier metrics have been generated yet.", flush=True)
                # Continue with other models. Do not fabricate results.
                continue
            finally:
                if rows:
                    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
                    print(f"  Saved comparison metrics: {output}", flush=True)
        if rows:
            print("\nFINAL RESULTS (mean ± sample SD):\n")
            print(pd.DataFrame(rows).to_string(index=False))
            print(f"\nResults file: {output}")
            if failed:
                print(f"WARNING: These checkpoints did NOT complete: {', '.join(failed)}")
                raise SystemExit(2)
        else:
            raise SystemExit("No models completed. Check the access/authentication and data paths above.")


if __name__ == "__main__":
    main()
