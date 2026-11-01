from pathlib import Path
import hashlib
import json
import pandas as pd

folder = Path(__file__).resolve().parent
source = folder / "inputs" / "candidate_827_matrix_deduplicated_percent.tsv"
expected_hash = "8591fe7ece89817e7923c0d1576d4f27a96b760ded4feef97b2f5770f607fa87"
if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
    raise ValueError("The source abundance matrix is not the verified candidate")

data = pd.read_csv(source, sep="\t", index_col=0, float_precision="round_trip")
metadata = pd.read_csv(folder / "inputs" / "metadata_389.tsv", sep="\t", index_col=0)
if data.shape != (389, 827) or not data.index.is_unique or not data.columns.is_unique:
    raise ValueError("Unexpected source matrix dimensions or identifiers")
if not metadata.index.is_unique or set(metadata.index) != set(data.index):
    raise ValueError("Metadata sample identifiers do not match the matrix")
metadata = metadata.loc[data.index]
metadata["Phenotype"] = metadata["Label"].map({1: "ASD", 0: "Healthy"})
if data.isna().any().any() or data.lt(0).any().any() or metadata.Phenotype.isna().any():
    raise ValueError("Missing/invalid abundance or phenotype")

# Exclude samples only when all 827 species abundances equal zero.
exclude = data.eq(0).all(axis=1)
retained = data.loc[~exclude].copy()
included_metadata = metadata.loc[retained.index].copy()
excluded_metadata = metadata.loc[exclude].copy()
excluded_metadata["Exclusion_reason"] = "All 827 retained species abundances are zero"
if retained.shape != (251, 827) or int(exclude.sum()) != 138:
    raise ValueError("Unexpected cohort sizes")
if not retained.gt(0).any(axis=0).all():
    raise ValueError("A species has no positive abundance in the retained cohort")
if included_metadata.Phenotype.value_counts().to_dict() != {"Healthy": 128, "ASD": 123}:
    raise ValueError("Unexpected retained class counts")
if set(included_metadata["Project ID"]) != {"PRJNA1037036", "PRJNA686821"}:
    raise ValueError("Unexpected retained projects")
if set(excluded_metadata["Project ID"]) != {"PRJNA451479", "PRJNA516054"}:
    raise ValueError("Unexpected excluded projects")

output = folder / "results"
output.mkdir(exist_ok=True)
retained.to_csv(output / "species_251x827_deduplicated_percent.tsv", sep="\t")
included_metadata.to_csv(output / "metadata_251.tsv", sep="\t")
excluded_metadata.to_csv(output / "excluded_138_samples.tsv", sep="\t")
coverage = included_metadata.groupby(["Project ID", "Phenotype"]).size().unstack(fill_value=0)
coverage["Total"] = coverage.sum(axis=1)
coverage.to_csv(output / "project_class_counts.tsv", sep="\t")

# Zero counts and percentages for each sample and each species.
for axis, name in [(1, "by_sample"), (0, "by_species")]:
    counts = pd.DataFrame({"zero_count": retained.eq(0).sum(axis=axis),
                           "positive_count": retained.gt(0).sum(axis=axis),
                           "missing_NA": retained.isna().sum(axis=axis)})
    counts["total_positions"] = retained.shape[1] if axis == 1 else retained.shape[0]
    counts["zero_percent"] = 100 * counts.zero_count / counts.total_positions
    counts.index.name = "Run ID" if axis == 1 else "Feature"
    counts.to_csv(output / ("zero_counts_" + name + ".tsv"), sep="\t")

summary = {"samples": 251, "features": 827, "ASD": 123, "Healthy": 128,
           "projects": 2, "excluded_samples": 138, "abundance_units": "percent",
           "missing_NA": int(retained.isna().sum().sum()),
           "zero_cells": int(retained.eq(0).sum().sum()),
           "total_cells": retained.size,
           "zero_percent": 100 * int(retained.eq(0).sum().sum()) / retained.size,
           "all_zero_rows": int(retained.eq(0).all(axis=1).sum()),
           "all_zero_columns": int(retained.eq(0).all(axis=0).sum()),
           "source_matrix_sha256": expected_hash,
           "cohort_scope": "Derived two-project cohort; historical four-project cohort preserved",
           "duplicate_rule": "Retain one identical within-run abundance per original species TaxID",
           "model_trained": False}
(output / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(coverage.to_string())
print(json.dumps(summary, indent=2))
