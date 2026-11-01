from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PRED_FILE = Path(
    "GMrepo_ASD_Objective1/"
    "17_COMBINED_BASELINE_REVIEW/"
    "02_all_outer_project_predictions.tsv"
)

df = pd.read_csv(PRED_FILE, sep="\t")

# True label: 1 = ASD, 0 = Healthy
true_col = "Label"

# Find saved predicted class
pred_candidates = [
    "predicted_label",
    "Predicted_Label",
    "prediction",
    "Prediction",
    "y_pred",
]

pred_col = next(
    (c for c in pred_candidates if c in df.columns),
    None,
)

# If predicted class was not saved, reconstruct it using
# the SAVED model-specific threshold.
if pred_col is None:
    prob_candidates = [
        "probability",
        "predicted_probability",
        "ASD_probability",
        "y_prob",
        "prob",
    ]
    threshold_candidates = [
        "threshold",
        "selected_threshold",
        "classification_threshold",
    ]

    prob_col = next(
        (c for c in prob_candidates if c in df.columns),
        None,
    )
    threshold_col = next(
        (c for c in threshold_candidates if c in df.columns),
        None,
    )

    if prob_col is None or threshold_col is None:
        raise RuntimeError(
            "Could not find predicted-label columns or "
            "probability + saved-threshold columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    df["_predicted_label"] = (
        pd.to_numeric(df[prob_col], errors="raise")
        >= pd.to_numeric(df[threshold_col], errors="raise")
    ).astype(int)

    pred_col = "_predicted_label"


df[true_col] = pd.to_numeric(
    df[true_col], errors="raise"
).astype(int)

df[pred_col] = pd.to_numeric(
    df[pred_col], errors="raise"
).astype(int)


rows = []

for (feature_space, model), sub in df.groupby(
    ["feature_space", "model"],
    sort=False,
):
    y_true = sub[true_col]
    y_pred = sub[pred_col]

    rows.append(
        {
            "Feature space": feature_space,
            "Model": model,
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
        }
    )

results = pd.DataFrame(rows)

# Sort best accuracy first
results = results.sort_values(
    ["Accuracy", "F1"],
    ascending=False,
).reset_index(drop=True)

# Print as percentages
display = results.copy()
display["Accuracy"] = (
    display["Accuracy"] * 100
).map(lambda x: f"{x:.2f}%")
display["F1"] = (
    display["F1"] * 100
).map(lambda x: f"{x:.2f}%")

print("\nBASELINE CLASSIFICATION RESULTS")
print("=" * 75)
print(display.to_string(index=False))
print("=" * 75)

# Save exact decimal values
results.to_csv(
    "GMrepo_ASD_Objective1/"
    "17_COMBINED_BASELINE_REVIEW/"
    "08_accuracy_f1_summary.tsv",
    sep="\t",
    index=False,
)