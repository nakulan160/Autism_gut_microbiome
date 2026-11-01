from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time

# Limit supporting CPU work. The four fitted ML models use CUDA.
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "10"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/mnt/data/shyam/nakulan/09_10")
SEED = 42
METRICS = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1-Score"]
DATA_HASHES = {
    "species_251x827_deduplicated_percent.tsv": "272584a9b335eeaec13e97bd1b4d4c05713e09b20c9f18b7a56b409cf3834e24",
    "metadata_251.tsv": "fc707b997688369066b401cfa218c77bd80288114ed392382b1deaa4c081e21a",
    "FROZEN_SPECIES_VOCABULARY.tsv": "bfa319fc5b78b891fae046938b7a0675cfd73a7e696706831d351c4b30e767ac",
}


def load_data(root):
    # These hashes prevent accidentally using the old 389-row or summed matrix.
    for name, expected in DATA_HASHES.items():
        file = root / "data" / name
        if not file.is_file():
            raise FileNotFoundError("Move the supplied file here: " + str(file))
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError("Wrong or changed input file: " + str(file))
    matrix = pd.read_csv(root / "data" / "species_251x827_deduplicated_percent.tsv",
                         sep="\t", index_col=0, float_precision="round_trip")
    metadata = pd.read_csv(root / "data" / "metadata_251.tsv", sep="\t", index_col=0)
    vocabulary = pd.read_csv(root / "data" / "FROZEN_SPECIES_VOCABULARY.tsv", sep="\t")
    if matrix.shape != (251, 827) or not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("Expected exactly 251 unique samples and 827 unique species columns")
    if not metadata.index.is_unique or set(metadata.index) != set(matrix.index):
        raise ValueError("Matrix and metadata sample IDs do not match")
    metadata = metadata.loc[matrix.index]
    if list(matrix.columns) != vocabulary["Feature_Header"].tolist():
        raise ValueError("Species column order disagrees with the frozen vocabulary")
    if vocabulary["Species_ID"].tolist() != list(range(827)):
        raise ValueError("Unexpected species dictionary order")
    x = matrix.to_numpy(dtype=np.float64)
    labels = pd.to_numeric(metadata["Label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be exactly 0=Healthy or 1=ASD")
    y = labels.astype(np.int64)
    if np.bincount(y, minlength=2).tolist() != [128, 123]:
        raise ValueError("Expected 128 Healthy and 123 ASD samples")
    if not np.isfinite(x).all() or (x < 0).any() or (x > 100).any():
        raise ValueError("Invalid percent abundance")
    if (x == 0).all(axis=1).any() or not (x > 0).any(axis=0).all():
        raise ValueError("Unexpected all-zero sample or species")
    return x, y, metadata, vocabulary


def make_folds(x, y, metadata, root):
    # Both scripts call this function, so all six models use identical test folds.
    folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, y))
    assigned = np.zeros(len(y), dtype=int)
    for number, (train, test) in enumerate(folds, 1):
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("A fold lacks one class")
        assigned[test] = number
    manifest = pd.DataFrame({"Run ID": metadata.index, "Fold": assigned, "Label": y})
    path = root / "outputs" / "cv_folds.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        if not old.equals(manifest):
            raise ValueError("Existing cv_folds.tsv differs from the requested 10-fold protocol")
    else:
        manifest.to_csv(path, sep="\t", index=False)
    return folds


def calculate_metrics(y, predicted, score):
    predicted = np.asarray(predicted, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(predicted) != len(y) or len(score) != len(y) or not np.isfinite(score).all():
        raise ValueError("Invalid prediction shape or non-finite scores")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"Accuracy": float(accuracy_score(y, predicted)),
            "AUC": float(roc_auc_score(y, score)),
            "Sensitivity": float(tp / (tp + fn)),
            "Specificity": float(tn / (tn + fp)),
            "Precision": float(precision_score(y, predicted, zero_division=0)),
            "F1-Score": float(f1_score(y, predicted, zero_division=0)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def save_summary(rows, output, filename="mean_sd.tsv", expected_folds=10):
    table = pd.DataFrame(rows)
    summary = []
    for name, group in table.groupby("Model", sort=False):
        if len(group) != expected_folds or group["Fold"].nunique() != expected_folds:
            raise ValueError("Incomplete folds for " + name)
        result = {"Model": name, "Folds": len(group)}
        for metric in METRICS:
            result[metric + "_mean"] = float(group[metric].mean())
            result[metric + "_SD"] = float(group[metric].std(ddof=1))
        summary.append(result)
    numeric = pd.DataFrame(summary)
    numeric.to_csv(output / filename, sep="\t", index=False)
    display = pd.DataFrame({"Model": numeric["Model"]})
    for metric in METRICS:
        display[metric] = [f"{r[metric + '_mean']:.4f} ± {r[metric + '_SD']:.4f}"
                           for r in summary]
    display.to_csv(output / filename.replace(".tsv", "_formatted.tsv"), sep="\t", index=False)
    print(display.to_string(index=False), flush=True)


def versions():
    result = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "cuml", "cuml-cu12", "cuml-cu13", "cupy-cuda12x", "cupy-cuda13x"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def gpu_libraries(gpu):
    try:
        import cupy as cp
        import cuml
        import xgboost
        from cuml.ensemble import RandomForestClassifier
        from cuml.linear_model import LogisticRegression
        from cuml.preprocessing import StandardScaler
        from cuml.svm import SVC
    except ImportError as error:
        raise RuntimeError("GPU ML requires CuPy, RAPIDS cuML, and XGBoost >=2.0. "
                           "Install builds matching the server's NVIDIA driver/CUDA. "
                           "No CPU fallback is used.") from error
    if int(xgboost.__version__.split(".")[0]) < 2:
        raise RuntimeError("XGBoost >=2.0 is required for device='cuda'")
    if gpu < 0 or gpu >= cp.cuda.runtime.getDeviceCount():
        raise RuntimeError("Requested CUDA GPU is not available")
    cp.cuda.Device(gpu).use()
    cp.cuda.runtime.deviceSynchronize()
    info = cp.cuda.runtime.getDeviceProperties(gpu)
    print("CUDA GPU:", info["name"], "| cuML:", cuml.__version__, "| XGBoost:", xgboost.__version__, flush=True)

    def build(name, seed):
        if name == "XGBoost":
            return xgboost.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                         objective="binary:logistic", eval_metric="logloss",
                                         tree_method="hist", device=f"cuda:{gpu}",
                                         random_state=seed, n_jobs=10, verbosity=1)
        if name == "SVM":
            return SVC(C=1.0, kernel="rbf", gamma="scale", output_type="cupy")
        if name == "Random Forest":
            # Record the installed cuML defaults in run_config.json.
            # GPU quantile splits differ from scikit-learn's exact split algorithm.
            return RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                          random_state=seed, n_streams=1, output_type="cupy")
        return LogisticRegression(C=1.0, penalty="l2", max_iter=5000, output_type="cupy")
    return cp, build, StandardScaler


def main():
    parser = argparse.ArgumentParser(description="CUDA ML baselines on the exact 251×827 species cohort")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, default=0, help="GPU ordinal within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run-name", default="ml_gpu")
    parser.add_argument("--check-only", action="store_true", help="Verify data, folds and CUDA dependencies; do not fit models")
    args = parser.parse_args()
    x, y, metadata, vocabulary = load_data(args.root)
    folds = make_folds(x, y, metadata, args.root)
    cp, build, Scaler = gpu_libraries(args.gpu)
    if args.check_only:
        for name in ["XGBoost", "SVM", "Random Forest", "Logistic Regression"]:
            build(name, SEED).get_params()
        Scaler(output_type="cupy")
        print("Verified 251×827 inputs, class counts, ten shared folds and CUDA dependencies.")
        return
    output = args.root / "outputs" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    names = ["XGBoost", "SVM", "Random Forest", "Logistic Regression"]
    config = {"seed": SEED, "cv": "StratifiedKFold(10,shuffle=True,random_state=42)",
              "units": "percent", "class_labels": {"0": "Healthy", "1": "ASD"},
              "GPU": args.gpu, "versions": versions(), "input_sha256": DATA_HASHES,
              "scaling": "train-fold StandardScaler for SVM/LR only",
              "SD": "sample SD across ten test-fold metric values, ddof=1",
              "project_holdouts": False, "models": {}}
    for name in names:
        config["models"][name] = build(name, SEED).get_params()
    (output / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    all_rows, predictions = [], []
    x_gpu = cp.asarray(x, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.int32)
    for name in names:
        for fold, (train, test) in enumerate(folds, 1):
            start = time.perf_counter()
            model = build(name, SEED)
            train_x, test_x = x_gpu[train], x_gpu[test]
            if name in ["SVM", "Logistic Regression"]:
                scaler = Scaler(output_type="cupy")
                train_x = scaler.fit_transform(train_x)
                test_x = scaler.transform(test_x)
            model.fit(train_x, y_gpu[train])
            if name == "XGBoost":
                effective = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
                if not effective.startswith("cuda"):
                    raise RuntimeError("XGBoost did not train on CUDA: " + effective)
            classes = cp.asnumpy(cp.asarray(model.classes_)).astype(int).tolist()
            if classes != [0, 1]:
                raise ValueError("Unexpected model class ordering")
            predicted = cp.asnumpy(cp.asarray(model.predict(test_x))).astype(int).reshape(-1)
            if name == "SVM":
                score = cp.asnumpy(cp.asarray(model.decision_function(test_x))).reshape(-1)
                score_type = "decision_function_for_ASD"
            else:
                score = cp.asnumpy(cp.asarray(model.predict_proba(test_x)))[:, 1]
                score_type = "ASD_probability"
            result = calculate_metrics(y[test], predicted, score)
            all_rows.append({"Model": name, "Fold": fold, "Train_N": len(train), "Test_N": len(test),
                             "Seconds": time.perf_counter() - start, **result})
            predictions.extend({"Model": name, "Run ID": metadata.index[i], "Fold": fold,
                                "Label": int(y[i]), "Predicted": int(predicted[j]),
                                "ASD_score": float(score[j]), "Score_type": score_type}
                               for j, i in enumerate(test))
            pd.DataFrame(all_rows).to_csv(output / "fold_metrics.tsv", sep="\t", index=False)
            pd.DataFrame(predictions).to_csv(output / "oof_predictions.tsv", sep="\t", index=False)
            print(name, f"fold {fold}/10:", ", ".join(f"{m}={result[m]:.4f}" for m in METRICS), flush=True)
            del model, train_x, test_x
        save_summary([r for r in all_rows if r["Model"] == name], output,
                     filename=name.replace(" ", "_") + "_mean_sd.tsv")
    save_summary(all_rows, output)
    pooled = []
    for name, group in pd.DataFrame(predictions).groupby("Model", sort=False):
        if len(group) != 251 or not group["Run ID"].is_unique:
            raise ValueError("Incomplete or duplicate out-of-fold predictions")
        pooled.append({"Model": name, **calculate_metrics(group.Label.to_numpy(), group.Predicted.to_numpy(), group.ASD_score.to_numpy())})
    pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.tsv", sep="\t", index=False)
    (output / "COMPLETED").write_text("Four CUDA models, ten folds each; no project-held-out evaluation.\n")


if __name__ == "__main__":
    main()
