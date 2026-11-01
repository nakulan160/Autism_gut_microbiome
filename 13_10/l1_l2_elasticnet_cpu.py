#!/usr/bin/env python3
"""ASD vs Healthy: 10-fold L1/L2/Elastic Net feature selection + baseline classifiers.

Input: original species-model input matrix; output: one 12-row TSV table.
No hyperparameter searches. Feature selectors fit only on outer-training folds.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "10"

from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, f1_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier  # XGBoost is not part of sklearn; uses its sklearn-compatible API.

SEED = 42
MATRIX_SHA256 = "32e8bcbd472258c6886f7f4df5e6f7d56060f4bff423a4487cc956343f6135fd"
HERE = Path(__file__).resolve().parent
INPUT_CANDIDATES = ("Species_model_input_matrix.tsv", "species_model_input_matrix.tsv",
                    "species_model_input_matrix(1).tsv")
DATA_FILE = next((HERE / name for name in INPUT_CANDIDATES if (HERE / name).exists()), None)
OUTPUT_FILE = HERE / "l1_l2_elasticnet_cpu_results.tsv"
METRICS = ("Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1")
METHODS = ("L1", "L2 (matched L1 count)", "Elastic Net (0.5 L1 + 0.5 L2)")
MODELS = ("XGBoost", "Logistic Regression", "Random Forest", "SVM (RBF)")

ASD_RUNS = """
SRR10338792 SRR10338793 SRR10338794 SRR10338795 SRR10338796 SRR10338797 SRR10338798 SRR10338799
SRR10338800 SRR10338804 SRR10338805 SRR10338806 SRR10338807 SRR10338808 SRR10338809 SRR10338810
SRR10338811 SRR10338812 SRR10338813 SRR10338814 SRR10338815 SRR10338816 SRR10338817 SRR10338818
SRR10338819 SRR13327429 SRR13327440 SRR13327447 SRR13327448 SRR13327449 SRR13327450 SRR13327451
SRR13327452 SRR13327453 SRR13327454 SRR13327455 SRR13327456 SRR13327457 SRR13327458 SRR13327459
SRR13327460 SRR13327461 SRR13327462 SRR13327463 SRR13327464 SRR13327465 SRR13327466 SRR13327467
SRR13327468 SRR13327469 SRR13327470 SRR13327471 SRR13327472 SRR13327473 SRR13327474 SRR13327475
SRR13327476 SRR13327477 SRR13327478 SRR13327479 SRR13327480 SRR13327481 SRR13327482 SRR13327483
SRR13327484 SRR13327485 SRR13327486 SRR13327487 SRR13327488 SRR13327489 SRR13327490 SRR13327491
SRR13327492 SRR13327493 SRR13327494 SRR13327495 SRR13327496 SRR13327497 SRR13327498 SRR13327499
SRR13327500 SRR13327501 SRR13327502 SRR13327511 SRR13327522 SRR13327533 SRR13327534 SRR13327535
SRR13327546 SRR26718071 SRR26718082 SRR26718093 SRR26718104 SRR26718105 SRR26718106 SRR26718107
SRR26718108 SRR26718109 SRR26718110 SRR26718111 SRR26718112 SRR26718113 SRR26718114 SRR26718115
SRR26718116 SRR26718117 SRR26718118 SRR26718119 SRR26718120 SRR26718121 SRR26718122 SRR26718123
SRR26718124 SRR26718125 SRR26718126 SRR26718127 SRR26718128 SRR26718129 SRR26718130 SRR26718131
SRR26718132 SRR26718133 SRR26718134 SRR26718135 SRR26718136 SRR26718137 SRR26718138 SRR26718139
SRR26718140 SRR26718141 SRR26718142 SRR26718143 SRR26718144 SRR26718145 SRR26718146 SRR26718147
SRR26718148 SRR26718149 SRR26718150 SRR26718151 SRR26718152 SRR26718153 SRR26718154 SRR26718160
SRR26718171 SRR26718182 SRR26718183 SRR26718184 SRR7057650 SRR7057651 SRR7057652 SRR7057653
SRR7057654 SRR7057655 SRR7057656 SRR7057657 SRR7057658 SRR7057659 SRR7057660 SRR7057661
SRR7057662 SRR7057663 SRR7057664 SRR7057665 SRR7057666 SRR7057667 SRR7057668 SRR7057669
SRR7057670 SRR7057671 SRR7057672 SRR7057673 SRR7057674 SRR7057675 SRR7057676 SRR7057677
SRR7057678 SRR7057679 SRR8490384 SRR8490385 SRR8490386 SRR8490387 SRR8490388 SRR8490389
SRR8490390 SRR8490391 SRR8490392 SRR8490393 SRR8490406 SRR8490407 SRR8490408 SRR8490409
SRR8490411 SRR8524508 SRR8524510 SRR8524511 SRR8524512 SRR8524513 SRR8524514 SRR8524515
SRR8524516 SRR8524517 SRR8524518 SRR8524519 SRR8524520 SRR8524523 SRR8524525 SRR8524526
""".split()
HEALTHY_RUNS = """
SRR10338801 SRR10338802 SRR10338803 SRR13327423 SRR13327424 SRR13327425 SRR13327426 SRR13327427
SRR13327428 SRR13327430 SRR13327431 SRR13327432 SRR13327433 SRR13327434 SRR13327435 SRR13327436
SRR13327437 SRR13327438 SRR13327439 SRR13327441 SRR13327442 SRR13327443 SRR13327444 SRR13327445
SRR13327446 SRR13327503 SRR13327504 SRR13327505 SRR13327506 SRR13327507 SRR13327508 SRR13327509
SRR13327510 SRR13327512 SRR13327513 SRR13327514 SRR13327515 SRR13327516 SRR13327517 SRR13327518
SRR13327519 SRR13327520 SRR13327521 SRR13327523 SRR13327524 SRR13327525 SRR13327526 SRR13327527
SRR13327528 SRR13327529 SRR13327530 SRR13327531 SRR13327532 SRR13327536 SRR13327537 SRR13327538
SRR13327539 SRR13327540 SRR13327541 SRR13327542 SRR13327543 SRR13327544 SRR13327545 SRR13327547
SRR13327548 SRR13327549 SRR13327550 SRR26718061 SRR26718062 SRR26718063 SRR26718064 SRR26718065
SRR26718066 SRR26718067 SRR26718068 SRR26718069 SRR26718070 SRR26718072 SRR26718073 SRR26718074
SRR26718075 SRR26718076 SRR26718077 SRR26718078 SRR26718079 SRR26718080 SRR26718081 SRR26718083
SRR26718084 SRR26718085 SRR26718086 SRR26718087 SRR26718088 SRR26718089 SRR26718090 SRR26718091
SRR26718092 SRR26718094 SRR26718095 SRR26718096 SRR26718097 SRR26718098 SRR26718099 SRR26718100
SRR26718101 SRR26718102 SRR26718103 SRR26718156 SRR26718157 SRR26718158 SRR26718159 SRR26718161
SRR26718162 SRR26718163 SRR26718164 SRR26718165 SRR26718166 SRR26718167 SRR26718168 SRR26718169
SRR26718170 SRR26718172 SRR26718173 SRR26718174 SRR26718175 SRR26718176 SRR26718177 SRR26718178
SRR26718179 SRR26718180 SRR26718181 SRR7057620 SRR7057621 SRR7057622 SRR7057623 SRR7057624
SRR7057625 SRR7057626 SRR7057627 SRR7057628 SRR7057629 SRR7057630 SRR7057631 SRR7057632
SRR7057633 SRR7057634 SRR7057635 SRR7057636 SRR7057637 SRR7057638 SRR7057639 SRR7057640
SRR7057641 SRR7057642 SRR7057643 SRR7057644 SRR7057645 SRR7057646 SRR7057647 SRR7057648
SRR7057649 SRR8490394 SRR8490395 SRR8490396 SRR8490397 SRR8490398 SRR8490399 SRR8490400
SRR8490401 SRR8490402 SRR8490403 SRR8490404 SRR8490405 SRR8490410 SRR8490412 SRR8490413
SRR8524509 SRR8524521 SRR8524522 SRR8524524 SRR8524527
""".split()


def load_matrix():
    if DATA_FILE is None:
        raise FileNotFoundError(f"Put species_model_input_matrix.tsv beside {Path(__file__).name}")
    if hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() != MATRIX_SHA256:
        raise ValueError("Input is not the verified original 389-run species matrix.")
    df = pd.read_csv(DATA_FILE, sep="\t", dtype={"Run ID": str})
    ids = df["Run ID"].tolist()
    labels = {run: 1 for run in ASD_RUNS}
    labels.update({run: 0 for run in HEALTHY_RUNS})
    if len(ids) != 389 or len(set(ids)) != 389 or set(ids) != set(labels):
        raise ValueError("Run IDs do not match the original 208 ASD / 181 Healthy labels.")
    X = df.drop(columns="Run ID").to_numpy(dtype=np.float64)
    if X.shape != (389, 1068) or not np.isfinite(X).all():
        raise ValueError("Unexpected input matrix shape or missing values.")
    return X, np.array([labels[i] for i in ids])


def selected_columns(X, y):
    # All use sklearn's default regularization strength C=1. No CV tuning.
    # Solver is specified only because default 'lbfgs' cannot fit L1/Elastic Net.
    # max_iter=5000 only supports numerical convergence (as in the baseline script).
    selectors = {
        "l1": LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000, random_state=SEED),
        "l2": LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=SEED),
        "enet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                    C=1.0, max_iter=5000, random_state=SEED),
    }
    coefs = {}
    for name, model in selectors.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message="Inconsistent values: penalty=", category=UserWarning)
            model.fit(X, y)
        if model.n_iter_[0] >= 5000:
            raise RuntimeError(f"{name} feature selector did not converge.")
        coefs[name] = model.coef_[0]

    l1 = np.flatnonzero(coefs["l1"] != 0)
    if len(l1) == 0:
        raise RuntimeError("Default L1 selected zero features.")
    # L2 shrinks coefficients but does not force zeros; match L1's feature count.
    l2 = np.sort(np.argsort(-np.abs(coefs["l2"]), kind="stable")[:len(l1)])
    enet = np.flatnonzero(coefs["enet"] != 0)
    if len(enet) == 0:
        raise RuntimeError("Default Elastic Net selected zero features.")
    return {METHODS[0]: l1, METHODS[1]: l2, METHODS[2]: enet}


def models():
    return {
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  random_state=SEED, n_jobs=10, verbosity=0),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=10),
        "SVM (RBF)": SVC(),
    }


def metrics(y, predicted, scores):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, predicted), AUC=roc_auc_score(y, scores),
                Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp),
                Precision=precision_score(y, predicted, zero_division=0),
                F1=f1_score(y, predicted, zero_division=0))


def main():
    X, y = load_matrix()
    records = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    print(f"Verified: {X.shape[0]} samples, {X.shape[1]} taxa; ASD={sum(y == 1)}, Healthy={sum(y == 0)}")
    print("L1, L2 and Elastic Net; default regularization C=1; no hyperparameter tuning.")
    for fold, (train, test) in enumerate(cv.split(X, y), 1):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        scaler = StandardScaler().fit(Xtrain)
        scaled_train = scaler.transform(Xtrain)
        scaled_test = scaler.transform(Xtest)
        choices = selected_columns(scaled_train, ytrain)
        print(f"Fold {fold}/10: " + ", ".join(f"{k}={len(v)}" for k, v in choices.items()), flush=True)
        for method, columns in choices.items():
            for name, model in models().items():
                # Same baseline behaviour: raw abundances for trees, scaled for LR and SVM.
                a, b = ((Xtrain, Xtest) if name in ("XGBoost", "Random Forest")
                        else (scaled_train, scaled_test))
                model.fit(a[:, columns], ytrain)
                predicted = model.predict(b[:, columns])
                scores = (model.decision_function(b[:, columns]) if name == "SVM (RBF)"
                          else model.predict_proba(b[:, columns])[:, 1])
                records.append({"Method": method, "Model": name, "Features": len(columns),
                                **metrics(ytest, predicted, scores)})
    rows = []
    for method in METHODS:
        for name in MODELS:
            sub = [r for r in records if r["Method"] == method and r["Model"] == name]
            format_ms = lambda vals, digits: f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"
            row = {"Method": method, "Model": name,
                   "Features (mean ± SD)": format_ms([r["Features"] for r in sub], 1)}
            for metric in METRICS:
                row[metric] = format_ms([r[metric] for r in sub], 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved exactly one output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
