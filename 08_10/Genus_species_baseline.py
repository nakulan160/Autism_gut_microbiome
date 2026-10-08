from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


SEED = 42

HERE = Path(__file__).resolve().parent

DATA_DIR = (
    HERE.parent
    / "07_10"
    / "GMrepo_ASD_Objective1"
    / "1_results"
)

GENUS_FILE = DATA_DIR / "04_genus_abundance_long.tsv"
SPECIES_FILE = DATA_DIR / "03_species_abundance_long.tsv"
META_FILE = DATA_DIR / "06_sample_metadata_for_modelling.tsv"

GENUS_OUT = HERE / "Genus_baseline_results.tsv"
SPECIES_OUT = HERE / "Species_baseline_results.tsv"

GENUS_MATRIX_OUT = HERE / "Genus_model_input_matrix.tsv"
SPECIES_MATRIX_OUT = HERE / "Species_model_input_matrix.tsv"


def normalise_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(df, candidates):
    lookup = {normalise_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_data(abundance_file, taxonomic_level):
    abundance = pd.read_csv(
        abundance_file,
        sep="\t",
        low_memory=False,
    )

    meta = pd.read_csv(
        META_FILE,
        sep="\t",
        low_memory=False,
    )

    run_abundance = find_column(
        abundance,
        ["Run ID", "Run_ID", "run_id", "run"],
    )

    taxon_col = find_column(
        abundance,
        [
            "scientific_name",
            "taxon_name",
            "taxon",
            "name",
            taxonomic_level,
        ],
    )

    abundance_col = find_column(
        abundance,
        [
            "relative_abundance",
            "rel_abundance",
            "relative abundance",
            "abundance",
            "Relative abundance",
        ],
    )

    run_meta = find_column(
        meta,
        ["Run ID", "Run_ID", "run_id", "run"],
    )

    project_col = find_column(
        meta,
        ["Project ID", "Project_ID", "project_id", "project"],
    )

    label_col = find_column(
        meta,
        ["Label", "label"],
    )

    abundance = abundance[
        [run_abundance, taxon_col, abundance_col]
    ].copy()

    abundance.columns = [
        "Run ID",
        "Taxon",
        "Abundance",
    ]

    abundance["Run ID"] = (
        abundance["Run ID"]
        .astype(str)
        .str.strip()
    )

    abundance["Taxon"] = (
        abundance["Taxon"]
        .astype(str)
        .str.strip()
    )

    abundance["Abundance"] = pd.to_numeric(
        abundance["Abundance"],
        errors="coerce",
    )

    abundance = abundance.loc[
        abundance["Run ID"].ne("")
        & abundance["Taxon"].ne("")
        & abundance["Abundance"].notna()
    ].copy()

    matrix = (
        abundance
        .groupby(
            ["Run ID", "Taxon"],
            as_index=False,
        )["Abundance"]
        .sum()
        .pivot(
            index="Run ID",
            columns="Taxon",
            values="Abundance",
        )
        .fillna(0.0)
        .sort_index(axis=1)
    )

    meta = meta[
        [run_meta, project_col, label_col]
    ].copy()

    meta.columns = [
        "Run ID",
        "Project ID",
        "Label",
    ]

    meta["Run ID"] = (
        meta["Run ID"]
        .astype(str)
        .str.strip()
    )

    meta["Project ID"] = (
        meta["Project ID"]
        .astype(str)
        .str.strip()
    )

    meta["Label"] = pd.to_numeric(
        meta["Label"],
        errors="raise",
    ).astype(int)

    meta = (
        meta
        .drop_duplicates(subset=["Run ID"])
        .set_index("Run ID")
    )

    common_runs = matrix.index.intersection(meta.index)

    matrix = matrix.loc[common_runs].copy()
    meta = meta.loc[common_runs].copy()

    if not set(meta["Label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected labels to be 0=Healthy and 1=ASD."
        )

    x = matrix.to_numpy(dtype=float)
    y = meta["Label"].to_numpy()

    return x, y, meta, matrix


def build_models():
    return {
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),

        "SVM": Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    SVC(),
                ),
            ]
        ),

        "Logistic Regression": Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def get_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    return model.decision_function(x)


def calculate_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "AUC": roc_auc_score(
            y_true,
            y_score,
        ),

        "Sensitivity": sensitivity,

        "Specificity": specificity,

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),

        "F1-Score": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
    }


def run_ten_fold(x, y, models):
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=SEED,
    )

    rows = []

    for model_name, base_model in models.items():
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for train_idx, test_idx in cv.split(x, y):
            model = clone(base_model)

            model.fit(
                x[train_idx],
                y[train_idx],
            )

            y_pred = model.predict(
                x[test_idx]
            )

            y_score = get_score(
                model,
                x[test_idx],
            )

            y_true_all.extend(
                y[test_idx]
            )

            y_pred_all.extend(
                y_pred
            )

            y_score_all.extend(
                y_score
            )

        result = calculate_metrics(
            np.asarray(y_true_all),
            np.asarray(y_pred_all),
            np.asarray(y_score_all),
        )

        rows.append(
            {
                "Evaluation": "10-fold CV",
                "Test Project": "All samples",
                "Model": model_name,
                **result,
            }
        )

    return rows


def run_project_wise(x, y, meta, models):
    rows = []

    projects = meta[
        "Project ID"
    ].to_numpy()

    unique_projects = sorted(
        pd.unique(projects)
    )

    for test_project in unique_projects:
        test_mask = (
            projects == test_project
        )

        train_mask = ~test_mask

        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Training data for {test_project} "
                "does not contain both classes."
            )

        if len(np.unique(y_test)) != 2:
            raise ValueError(
                f"Test project {test_project} "
                "does not contain both classes, "
                "so AUC cannot be calculated."
            )

        for model_name, base_model in models.items():
            model = clone(base_model)

            model.fit(
                x[train_mask],
                y_train,
            )

            y_pred = model.predict(
                x[test_mask]
            )

            y_score = get_score(
                model,
                x[test_mask],
            )

            result = calculate_metrics(
                y_test,
                y_pred,
                y_score,
            )

            rows.append(
                {
                    "Evaluation": "Project-wise",
                    "Test Project": test_project,
                    "Model": model_name,
                    **result,
                }
            )

    return rows


def run_analysis(
    abundance_file,
    taxonomic_level,
    output_file,
    matrix_output_file,
):
    x, y, meta, matrix = load_data(
        abundance_file,
        taxonomic_level,
    )

    matrix.to_csv(
        matrix_output_file,
        sep="\t",
        index=True,
    )

    print(
        f"\n{taxonomic_level.upper()} INPUT MATRIX SHAPE: "
        f"{matrix.shape[0]} samples x {matrix.shape[1]} taxa"
    )

    print(
        f"\n{taxonomic_level.upper()} INPUT MATRIX - FIRST 5 SAMPLES"
    )

    print(
        matrix.iloc[:5, :10].to_string()
    )

    models = build_models()

    rows = run_ten_fold(
        x,
        y,
        models,
    )

    rows.extend(
        run_project_wise(
            x,
            y,
            meta,
            models,
        )
    )

    results = pd.DataFrame(rows)

    metric_columns = [
        "Accuracy",
        "AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1-Score",
    ]

    results[metric_columns] = (
        results[metric_columns]
        .round(4)
    )

    results.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(
        f"\n{taxonomic_level.upper()} - "
        "10-FOLD CROSS-VALIDATION"
    )

    print(
        results.loc[
            results["Evaluation"]
            == "10-fold CV",
            ["Model"] + metric_columns,
        ].to_string(index=False)
    )

    print(
        f"\n{taxonomic_level.upper()} - "
        "PROJECT-WISE TRAIN/TEST"
    )

    print(
        results.loc[
            results["Evaluation"]
            == "Project-wise",
            [
                "Test Project",
                "Model",
            ]
            + metric_columns,
        ].to_string(index=False)
    )


def main():
    required_files = [
        GENUS_FILE,
        SPECIES_FILE,
        META_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

    run_analysis(
        abundance_file=GENUS_FILE,
        taxonomic_level="Genus",
        output_file=GENUS_OUT,
        matrix_output_file=GENUS_MATRIX_OUT,
    )

    run_analysis(
        abundance_file=SPECIES_FILE,
        taxonomic_level="Species",
        output_file=SPECIES_OUT,
        matrix_output_file=SPECIES_MATRIX_OUT,
    )


if __name__ == "__main__":
    main()
