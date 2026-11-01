#!/usr/bin/env python3
"""No-tuning ASD vs healthy comparison: L1, L2, and Elastic Net; one TSV output."""
import os
for env in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[env] = '10'

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
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, f1_score)
from xgboost import XGBClassifier

SEED = 42
HERE = Path(__file__).resolve().parent
CANDIDATES = ('species_model_input_matrix.tsv', 'species_model_input_matrix(1).tsv',
              'Species_model_input_matrix.tsv')
EXPECTED_SHA = '32e8bcbd472258c6886f7f4df5e6f7d56060f4bff423a4487cc956343f6135fd'
OUT = HERE / 'regularization_default_cpu_results.tsv'
METRICS = ('Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1')
METHODS = ('L1', 'L2 (all features)', 'Elastic Net (L1 ratio 0.5)')
MODELS = ('XGBoost', 'Logistic Regression', 'Random Forest', 'SVM (RBF)')

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


def load_data():
    path = next((HERE / name for name in CANDIDATES if (HERE / name).is_file()), None)
    if path is None:
        raise FileNotFoundError('Put the original species_model_input_matrix.tsv alongside this script.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Matrix differs from the verified original file.')
    matrix = pd.read_csv(path, sep='\t', dtype={'Run ID': str})
    ids = matrix['Run ID'].tolist()
    label_map = {run: 1 for run in ASD_RUNS}
    label_map.update({run: 0 for run in HEALTHY_RUNS})
    if len(ids) != 389 or len(set(ids)) != 389 or set(ids) != set(label_map):
        raise ValueError('Sample IDs differ from the verified ASD/healthy labels.')
    X = matrix.drop(columns=['Run ID']).to_numpy(dtype=np.float64)
    y = np.asarray([label_map[run] for run in ids], dtype=int)
    if X.shape != (389, 1068) or not np.isfinite(X).all() or (X < 0).any():
        raise ValueError('Bad abundance matrix values/shape.')
    return X, y


def calculate(y, predicted, scores):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, predicted), AUC=roc_auc_score(y, scores),
                Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp),
                Precision=precision_score(y, predicted, zero_division=0),
                F1=f1_score(y, predicted, zero_division=0))


def model(name):
    if name == 'XGBoost':
        return XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                             random_state=SEED, n_jobs=10, verbosity=0)
    if name == 'Random Forest':
        return RandomForestClassifier(random_state=SEED, n_jobs=10)
    if name == 'SVM (RBF)':
        return SVC()
    raise ValueError(name)


def main():
    X, y = load_data()
    print(f'{X.shape[0]} runs; {X.shape[1]} taxa; ASD={(y == 1).sum()}, healthy={(y == 0).sum()}')
    print('Default C=1; L1/EN require alternative solvers; no search or threshold tuning.')
    results = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(cv.split(X, y), start=1):
        raw_train, raw_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scaler = StandardScaler().fit(raw_train)
        scaled_train, scaled_test = scaler.transform(raw_train), scaler.transform(raw_test)
        classifiers = (
            LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, random_state=SEED),
            LogisticRegression(penalty='l2', max_iter=5000, random_state=SEED),
            LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
                               solver='saga', max_iter=5000, random_state=SEED),
        )
        print(f'Fold {fold}/10:', end=' ', flush=True)
        for method, selector in zip(METHODS, classifiers):
            # These models do REAL L1/L2/ElasticNet regularization of the LR classifier.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                warnings.filterwarnings('ignore', message='Inconsistent values: penalty=', category=UserWarning)
                selector.fit(scaled_train, y_train)
            if selector.n_iter_[0] >= 5000:
                raise RuntimeError(f'{method} failed to converge in fold {fold}.')
            if method == METHODS[1]:
                # L2 is shrinkage, not feature deletion. Keeping all is the correct definition.
                columns = np.arange(X.shape[1])
            else:
                columns = np.flatnonzero(selector.coef_[0] != 0)
            if not len(columns):
                raise ValueError(f'Zero selected features: {method}, fold {fold}.')
            print(f'{method}={len(columns)}', end='; ', flush=True)
            for name in MODELS:
                if name == 'Logistic Regression':
                    predicted = selector.predict(scaled_test)
                    scores = selector.predict_proba(scaled_test)[:, 1]
                else:
                    estimator = model(name)
                    a, b = ((raw_train, raw_test) if name in ('XGBoost', 'Random Forest')
                            else (scaled_train, scaled_test))
                    estimator.fit(a[:, columns], y_train)
                    predicted = estimator.predict(b[:, columns])
                    scores = (estimator.decision_function(b[:, columns]) if name == 'SVM (RBF)'
                              else estimator.predict_proba(b[:, columns])[:, 1])
                results.append(dict(Method=method, Model=name, Features=len(columns),
                                    **calculate(y_test, predicted, scores)))
        print(flush=True)

    rows = []
    for method in METHODS:
        for name in MODELS:
            subset = [r for r in results if r['Method'] == method and r['Model'] == name]
            def fmt(key, precision=4):
                data = [r[key] for r in subset]
                return f'{np.mean(data):.{precision}f} ± {np.std(data, ddof=1):.{precision}f}'
            row = dict(Method=method, Model=name, **{'Features (mean ± SD)': fmt('Features', 1)})
            row.update({metric: fmt(metric) for metric in METRICS})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, sep='\t', index=False)
    print('\n' + summary.to_string(index=False))
    print(f'\nSaved exactly one output file: {OUT}')

if __name__ == '__main__':
    main()
