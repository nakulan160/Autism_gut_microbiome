# Derived 251-sample, 827-species GMrepo cohort

Following the user's instruction to use the remaining 251 samples, this
dataset excludes the 138 samples whose abundances are zero for all 827
retained species. It is a separate two-project analysis cohort. The original
389-sample files and their evidence are preserved.

| Retained project | ASD | Healthy | Total |
|---|---:|---:|---:|
| PRJNA1037036 | 59 | 64 | 123 |
| PRJNA686821 | 64 | 64 | 128 |
| Total | 123 | 128 | 251 |

Excluded: all 60 samples from PRJNA451479 and all 78 from PRJNA516054.
These retrieved GMrepo profiles lacked entries with NCBI rank exactly
species. This is not evidence that their biological samples contained no
species. The exclusions remove two whole studies and change the study scope.

## Files to use

- `results/species_251x827_deduplicated_percent.tsv`: abundance matrix.
  Its first column is Run ID; the following 827 are species abundance features.
- `results/metadata_251.tsv`: matched sample metadata and phenotypes.
  Label 1 is ASD and Label 0 is Healthy.
- `results/excluded_138_samples.tsv`: all excluded samples and reasons.
- `results/project_class_counts.tsv`: retained project and phenotype counts.
- `results/zero_counts_by_sample.tsv` and `zero_counts_by_species.tsv`:
  zero/missing/positive counts and zero percentages.
- `results/cohort_summary.json`: overall counts, zero percentage and provenance.
- `FROZEN_SPECIES_VOCABULARY.tsv`: the unchanged 827 species identities/order.
- `inputs/`: the verified 389-row candidate and original source-verified
  metadata, retained for reproducibility. These are inputs, not the new cohort.
- `input_and_result_sha256.json`: hashes for the data, vocabulary and script.

Run `python3 make_251_cohort.py` to reproduce the derived files offline.
Requires pandas. Paths are relative to the script, so no path editing is needed.

## What changed

Only the all-zero rows were excluded. All 827 species columns, their order,
retained sample abundance values, sample IDs and labels were preserved.
Abundances remain in percent; there was no rescaling, renormalization,
prevalence filter or new taxonomic mapping. Each of the 827 species has at
least one positive abundance among these 251 samples.

The source is the duplicate-corrected candidate from the species audit. Its
handling of repeated identical TaxID/abundance records is a documented
source-record assumption. Original raw reads were not independently profiled.

The historical XGBoost and Transformer results used other cohorts/input
definitions and are not results for this new dataset. No model was trained
while creating it. V2 currently checks for the old 389-sample/four-project
cohort and will need an explicit cohort update before use. The V1/V2 alignment
and evaluation protocol remain to be finalized.
