# Autism gut microbiome — project state and chat handover

**Project:** `autism_gut_microbiome`  
**Working title:** Understanding Gut Brain Axis in Autism Employing Deep Learning  
**Researcher in this project:** Nakulan (as named in the supplied export)  
**Updated:** 2026-10-09, Asia/Calcutta  
**Checkpoint:** v20 — user-selected derived 251×827 two-project cohort created; historical 389-row inputs preserved; no model trained.  
**Evidence cutoff:** Conversation export/objective files, recovered October 8 archives/acquisition code, supplied NCBI snapshot, and all 389 current GMrepo profiles retrieved and replayed on October 9.

## Read this first in the next chat

Continue this project from this document. Do not restart acquisition, change the objectives, invent results, or assume you can access every earlier chat. Read the latest checkpoint and the available scripts/output files before making changes. Preserve the historical frozen cohort, earlier results, taxonomy mappings, and evaluation designs. The historical strict-species matrix is not ready for new primary modeling; see the v18 audit below. Update this document after substantial verified progress.

The project has **three main objectives**: adapt existing microbiome foundation models to ASD; continue pretraining from existing pretrained weights; and pretrain a new microbiome foundation model from random initialization. Supplied files specify Objectives 1, 2, and 3; they do not contain a separate list of 23 objectives.

The historical frozen GMrepo cohort contains **389 samples: 208 ASD and 181 healthy, from four projects**. The genus matrix contains 332 features. The original GMrepo “species” matrix contains 1,068 mixed-rank taxonomic features. A corrected NCBI audit identified 829 species-ranked original features, which become **827 unique species-ranked TaxIDs after grouping**. The identity vocabulary is independently verified and frozen, but **138 runs in two projects have all-zero retained species abundance**, and **46 cells across 45 runs inherit duplicate source values**. In v20, the user selected a separate analysis excluding those 138 all-zero rows: **251 samples, 123 ASD and 128 Healthy, from two projects**, retaining all 827 species. All exclusions are documented; historical files are preserved.

Completed work reported in the export: acquisition and QC; frozen primary/sensitivity cohorts; nested project-aware genus ML baselines; frozen MGM representation evaluation and bootstrap comparisons; later genus/species default ML baselines using stratified 10-fold CV and project holdouts; MGM2 taxonomy compatibility investigation; and a supervised species Transformer V1.

Key results:

- Earlier nested project-held-out genus benchmark: best classical comparator, MGM-matched CLR Elastic Net, macro ROC-AUC **0.625527**; frozen mean-genus MGM + logistic probe, macro ROC-AUC **0.630086**. No established mean-AUC superiority; lower observed MGM project dispersion remains suggestive.
- Later 10-fold baseline: original species-profile XGBoost AUC **0.7911 ± 0.0551**, accuracy **0.7405 ± 0.0383**.
- Transformer V1 using 827 verified species: 10-fold AUC **0.5228 ± 0.1129**, accuracy **0.5217 ± 0.0298**; project holdouts often collapsed toward a single predicted class.

**Immediate task:** Teach and finalize the derived two-project 251×827 analysis protocol, then adapt V2's old cohort checks and align the recovered V1 implementation before a tokenization-only comparison. `gmrepo_251_species.zip` contains the new matrix, matched metadata, all 138 exclusions, frozen vocabulary, zero reports and reproducible script. The original `SPECIES_827_AUDIT_REPORT.md` and `species_827_evidence.zip` remain historical proof; their v18 `LOCK.json` still describes the blocked 389-row input and has not been rewritten. V2 code is prepared as `06_species_transformer_v2.py` but has not been run on the real cohort. Existing-model fine-tuning, representation/domain probes, and large-scale pretraining also remain pending. The **53,960-run master table is not proof of 53,960 downloaded/QC-approved microbiomes**.

## 1. Evidence and verification level

This file was assembled by extracting text from the 884-page `Nakulan_Deep Learning Gut Brain Axis.pdf` and inspecting relevant cohort, method, results, correction, and latest-planning passages. `pasted.txt` and `objectives.txt` were read in full. It is a structured research handover, not a verbatim export or a line-by-line certification of every old script.

No ASD model was rerun in this session. Model results below are **historical outputs/research-log results**, not newly replicated performance estimates. In v18, actual October 8 original/cleaned archives and acquisition code were recovered. Both species matrices, maps, the pinned taxdump, saved V1 outputs and V1 code were inspected, and all 389 current source profiles were independently checked. All 31 preserved non-code data/result files compared between the archives are byte-identical. The original acquisition raw cache, species long table and modeling metadata remain unavailable; metadata was recovered from two saved outputs and independently checked against current source responses. The earlier genus/MGM training and full acquisition/QC pipeline have not been rerun.

Source anchors refer to PDF page numbers, not Markdown line numbers. Important anchors:

| Source pages | What they document |
| --- | --- |
| 357–363 | Paired MGM/classical comparisons, robustness output, and generation of v16 |
| 364–385 | Scientific question, cohort, compatibility, frozen MGM method/results, interpretation |
| 395–396 | Correction: PRJEB23052 raw data versus missing processed profiles |
| 401–423 | Detailed acquisition, QC, frozen cohorts, matrices, and evaluation construction |
| 447–450 | Earlier pipeline cleanup and ordered script names |
| 602–603 | Corrected source paths inside `07_10/.../1_results/` |
| 764–770 | MGM2 setup, taxonomy paths, harmonization, and initial resolution audit |
| 780, 791–796 | Later ML fold mean/SD results and distinction from pooled metrics |
| 805, 813–814 | Corrected species-rank audit and final 827-token input |
| 838–855 | Detailed Transformer V1 research log, architecture, training, results, and files |
| 870–879 | V1 limitations, cleaned October 8 structure, and next experiment plan |

Earlier checkpoint `PROJECT_STATE_2026-10-07_v16.md` is named in the export on pages 363 and 395. The original v16 Markdown was located among project files during v18 retrieval but was not reread for this species audit. This handover began as a v17 reconstruction; it does not reproduce every v16 entry.

## 2. Scientific purpose and objectives

The central question is whether microbiome foundation models capture ASD-associated information that transfers across studies, or primarily capture project/domain structure. Prediction, study transfer, robustness, and reproducible biological interpretation must be considered together.

Taxonomic classification alone cannot establish a gut–brain mechanism or causality. Pathway, metabolite, immune, neural, and causal conclusions require additional data and analyses that are not completed in the supplied record.

### Objective 1 — Adapt existing microbiome foundation models for autism

Use pretrained models such as MGM and Waypoint, fine-tune or adapt them to ASD gut microbiome data, and evaluate their predictive performance, cross-study generalization, robustness, and biological relevance.

**Status:** Active. Classical baselines and initial frozen-MGM experiments reported complete. MGM2 compatibility investigated. Full foundation-model fine-tuning and completed Waypoint evaluation are not established in the supplied record.

### Objective 2 — Develop a microbiome foundation model using existing pretrained weights

Start with weights or representations from an established microbiome foundation model, continue large-scale pretraining or domain adaptation using a suitable microbiome corpus, and subsequently fine-tune and evaluate the adapted model.

**Status:** Planned. No completed continued-pretraining experiment documented. Corpus suitability and compatible model input/weights must be established from the data rather than assumed.

### Objective 3 — Develop a new microbiome foundation model from scratch

Design and pretrain a new foundation model starting from randomly initialized weights, using large-scale microbiome data to learn general microbial community representations before adapting it to ASD and other downstream tasks.

**Status:** Planned. Transformer V1 is a small supervised ASD classifier trained from scratch; it is not a pretrained foundation model and does not complete Objective 3.

## 3. Dataset provenance, retrieval, and frozen cohorts

Data source recorded in the project: GMrepo, <https://gmrepo.humangut.info/home>. The v18 species audit retrieved all 389 retained run profiles and preserved their raw responses; the broader acquisition universe was not rerun.

Initial inputs:

- `00_master_metagenomics_53960.tsv`: 53,960 shotgun/metagenomic runs in the master universe.
- `01_ASD_252.tsv`: 252 ASD runs from five projects.
- Same-project healthy controls produced 464 candidate runs, including 212 healthy controls.

Retrieval produced 389 usable profiles and 75 failures:

- All 74 PRJEB23052 runs (43 ASD, 31 controls) returned run metadata but zero species/genus profile rows. Later logs recorded `No species or genus profile returned`. This is a **processed-profile retrieval failure**, not evidence that the underlying raw sequencing data do not exist. The precise upstream GMrepo cause remains unresolved.
- `SRR26718155` from PRJNA1037036 also failed; the supplied history associates it with GMrepo QC=0.

Retained primary cohort:

| Project | ASD | Healthy | Total |
| --- | ---: | ---: | ---: |
| PRJNA1037036 | 59 | 64 | 123 |
| PRJNA451479 | 30 | 30 | 60 |
| PRJNA516054 | 55 | 23 | 78 |
| PRJNA686821 | 64 | 64 | 128 |
| **Total** | **208** | **181** | **389** |

All retained projects contain both classes. Use these as recorded sample/run counts; participant independence should be supported by metadata, not presumed solely from distinct run accessions.

QC examined successful retrieval, abundance scale/totals, richness, within-project richness outliers, exact profile hashes, possible repeated biological IDs, metadata completeness, and project class viability. Reported exact duplicate genus/species profiles: zero. This does not by itself prove absence of repeated participants.

GMrepo values were on percentage scale. Approximate sum range 95–105 and fewer than 10 positive taxa were **review flags**, not automatic exclusions.

Two low-richness ASD samples from PRJNA1037036 were retained in primary analysis:

| Run | Positive species | Positive genera |
| --- | ---: | ---: |
| SRR26718118 | 3 | 3 |
| SRR26718142 | 7 | 6 |

Frozen cohorts:

- **PRIMARY:** 389; 208 ASD, 181 healthy; includes both flagged samples.
- **SENSITIVITY:** 387; 206 ASD, 181 healthy; excludes those two runs.

Do not exclude samples because a later model performs poorly. A sensitivity cohort was defined; completion of all downstream sensitivity-model experiments is not established here.

Important historical outputs include `01_download_manifest.tsv`, `02_run_metadata.tsv`, `03_species_abundance_long.tsv`, `04_genus_abundance_long.tsv`, `06_sample_metadata_for_modelling.tsv`, `failed_or_unresolved_runs.tsv`, `08_QC/01_per_sample_qc.tsv`, `08_QC/10_qc_summary.json`, `09_FROZEN_COHORTS/02_PRIMARY_cohort.tsv`, and `09_FROZEN_COHORTS/03_SENSITIVITY_excluding_manual_QC_flags.tsv`.

## 4. Feature spaces and taxonomy corrections

| Representation | Recorded dimensions | Meaning |
| --- | --- | --- |
| Full genus matrix | 389 × 332 | Includes unresolved `Unknown` feature |
| Named genera | 389 × 331 | Excludes `Unknown` |
| MGM-matched genera | 389 × 227 | Genus vocabulary-compatible comparator |
| Original species-profile matrix | 389 × 1,068 | Mixed taxonomic ranks despite source filename |
| MGM2 harmonized taxonomy input | 389 × 1,063 before transpose | Five duplicate taxonomy mappings merged |
| Historical strict NCBI species Transformer input | 389 × 827 | 827 identities verified; 138 all-zero runs and duplicate abundance entries block new primary modeling |

Absent sample/taxon combinations are zeros. Do not silently apply top-K selection or an arbitrary abundance cutoff; the strict V1 preparation applied neither. Preserve units and row-sum audits for each method separately.

Corrected NCBI rank audit of the original 1,068 features:

| Rank | Count |
| --- | ---: |
| Species | 829 |
| Genus | 237 |
| Subspecies | 1 |
| Unresolved | 1 |
| **Total** | **1,068** |

The earlier MGM2 log reported 822 species, 233 genus, 1 subspecies, and 12 unresolved. That audit was subsequently corrected. Keep the earlier numbers only as an audit-history entry; use the corrected 829/237/1/1 counts for the current strict-species definition.

Historically, two pairs of source names were summed because they shared a current species TaxID, reducing 829 names to 827 identities. The v18 source audit shows that both pairs already shared the same original TaxID and abundance: these are name aliases. Identical repeated source rows for TaxID 84026 were also hidden within one original matrix column. A documented candidate deduplication changes 46 cells across 45 runs, retaining one identical value per run/source species TaxID. It has not been promoted to a primary analysis matrix. Frozen identities/order are in FROZEN_SPECIES_VOCABULARY.tsv inside the evidence package; do not remap silently.

**Comparison issue to resolve:** Later classical “species” baselines used the original 1,068-feature species-profile input, whereas V1 used 827 strict species. Their performance difference combines model and feature-space differences. Rerun relevant classical baselines on the exact 827-feature matrix for a controlled model comparison; preserve the original 1,068-feature baseline as a separate experiment.

## 5. Evaluation designs — keep the experiments separate

### Earlier nested cross-project genus/frozen-MGM benchmark

Four outer leave-one-project-out tests. Within each outer development set, the other three projects rotate as inner validation: 4 × 3 = 12 inner configurations, but **four outer test estimates**. Hyperparameters/thresholds are selected using development data only. Test project labels cannot influence preprocessing, tuning, or thresholds.

Macro project AUC means the equally weighted mean of the four held-out project AUCs. Bootstrap CIs resample samples within the observed projects and are conditional on those four studies; they do not quantify uncertainty across all possible future studies.

### Later default ML and Transformer benchmark

Stratified 10-fold CV, shuffle=True, random_state=42, plus separate held-out-project tests. V1 training used fixed settings and no hyperparameter tuning; later ML was also described as the default/no-tuning comparison following the guide's request.

Random folds mix project distributions and do not demonstrate transfer to a new study. Report them separately from project-held-out results. Do not compare 10-fold species XGBoost AUC 0.7911 directly with nested-LOPO frozen MGM macro AUC 0.6301 as if they share one protocol.

Mean ± SD across folds is different from a metric on pooled OOF predictions. In particular, uncalibrated SVM decision scores may have different scales across fitted folds. The later fold-wise species SVM AUC 0.6651 ± 0.0960 differs from the older pooled AUC 0.6318. Neither should be silently substituted for the other.

For continuation, verify a saved sample-to-fold manifest. The same seed alone does not establish identical splits if input sample order changed. Fit scaling, feature selection, learned transforms, and calibration within the appropriate training fold. Use biological subject grouping if repeated individuals are found.

## 6. Earlier MGM compatibility and frozen representation results

MGM recognized 227/332 genus names. `Unknown` is taxonomically unresolved abundance, not a biological genus. Recorded median raw abundance coverage was about 82.36%; median coverage among named taxa was 92.66%.

Named-genus median coverage by project: PRJNA1037036 97.35%; PRJNA451479 92.19%; PRJNA516054 93.51%; PRJNA686821 88.17%. In PRJNA516054, ASD minus healthy named coverage was about −3.76 percentage points, with diagnostic p≈0.0033. Treat that as possible model-interface/domain confounding, not a biological discovery.

Recorded decision: `PROCEED_WITH_MGM_WITH_CONTROLS`.

Frozen MGM primary extraction was corrected to **mean pooling of pretrained genus hidden states**, excluding BOS/EOS/padding. Label: `MGM_PRETRAINED_MEAN_GENUS`. Frozen EOS extraction remained a sensitivity comparison, not the primary representation. No MGM weights were trained; an L2 logistic probe used project-aware development folds. Reported input sequence lengths: minimum 5, maximum 79, average about 37.52 tokens; all 389 samples retained.

| Earlier experiment | Macro ROC-AUC | Other recorded metrics |
| --- | ---: | --- |
| MGM_MATCHED_227 + ElasticNet_CLR | 0.625527 | PR-AUC 0.6981; balanced accuracy 0.5318; sample-bootstrap AUC CI 0.5643–0.6834 |
| Full 332 + XGBoost | 0.613693 | Separate classical feature-space comparator |
| NAMED_331 + XGBoost | 0.611333 | Separate classical feature-space comparator |
| Technical coverage-only logistic | 0.556936 | Recorded CI crossed 0.5; this does not rule out other confounders |
| Frozen MGM mean-genus + logistic probe | 0.630086 | PR-AUC 0.6852; balanced accuracy 0.5617; sensitivity 0.4325; specificity 0.6909 |
| Frozen MGM EOS sensitivity | approximately 0.5567 | Poorer than mean-genus in this experiment |

Approximate project AUCs from the historical research log:

| Held-out project | Best matched Elastic Net | Frozen mean-genus MGM |
| --- | ---: | ---: |
| PRJNA1037036 | 0.783 | 0.612 |
| PRJNA451479 | 0.604 | 0.668 |
| PRJNA516054 | 0.482 | 0.606 |
| PRJNA686821 | 0.632 | 0.635 |

Use original prediction/result tables for additional decimals. MGM was not uniformly better: it lost performance on PRJNA1037036 and improved some other projects.

Paired, within-project/phenotype-stratified bootstrap comparisons against matched Elastic Net:

| Quantity | Observed | Recorded 95% CI / bootstrap support |
| --- | ---: | --- |
| Mean macro-AUC difference, initial comparison | +0.004559 | −0.059334 to +0.070078; proportion delta>0 = 0.5592 |
| Mean macro-AUC difference, robustness run | +0.004559 | −0.058396 to +0.068380 |
| Project AUC range reduction | +0.238911 | −0.015436 to +0.338363; proportion reduction>0 = 0.9650 |
| Worst-project improvement | +0.123320 | −0.035191 to +0.209495 |

The robustness run was reported with 10,000 bootstrap replicates. Frozen MGM observed project-range width was 0.062244 versus Elastic Net 0.301155. These data do **not** establish superior mean AUC or a statistically confirmed robustness advantage: the relevant CIs include zero. The 0.965 bootstrap fraction is not automatically a p-value of 0.035. Similar point estimates do not establish formal statistical equivalence either.

Unresolved: audit overlap with MGM pretraining; probe project identity, phenotype, coverage, and available covariates; test biological reproducibility and sensitivity cohort. Do not claim outer test projects were unseen during foundation-model pretraining until audited.

## 7. Later classical ML results

Reported mean ± SD across 10 stratified folds:

| Feature input | Model | Accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Original species-profile matrix | XGBoost | 0.7405 ± 0.0383 | 0.7911 ± 0.0551 |
| Original species-profile matrix | SVM | 0.6092 ± 0.0415 | 0.6651 ± 0.0960 |
| Original species-profile matrix | Logistic regression | 0.6605 ± 0.0354 | 0.6886 ± 0.0374 |
| Original species-profile matrix | Random forest | 0.6891 ± 0.0576 | 0.7417 ± 0.0636 |
| Genus | XGBoost | 0.6428 ± 0.0862 | 0.6871 ± 0.0964 |
| Genus | SVM | 0.6608 ± 0.0794 | 0.6954 ± 0.0984 |
| Genus | Logistic regression | 0.6169 ± 0.0492 | 0.6660 ± 0.0705 |
| Genus | Random forest | 0.6428 ± 0.1025 | 0.6979 ± 0.0837 |

Species-profile XGBoost additional metrics: sensitivity 0.7933 ± 0.0962; specificity 0.6792 ± 0.1049; precision 0.7447 ± 0.0437; F1 0.7639 ± 0.0415. Older pooled F1 0.7657 and manual `0.7657(0.005)` are not the fold mean/SD result.

Separate later project-held-out species-profile XGBoost results, as recorded in the export:

| Project | Accuracy | AUC |
| --- | ---: | ---: |
| PRJNA1037036 | 0.4797 | 0.7918 |
| PRJNA451479 | 0.4833 | 0.4844 |
| PRJNA516054 | 0.6026 | 0.5067 |
| PRJNA686821 | 0.7422 | 0.8508 |

These expose substantial project variation and differences between ranking quality and thresholded accuracy. They are a different benchmark from the earlier tuned nested genus analysis. Confirm original files and exact default settings before extending the comparison.

## 8. MGM2 investigation and corrections

MGM2-small was prepared in a separate environment on CPU. A pandas read-only-array issue in `build_dataset.py` was patched; processing of all 389 samples was reported successful. Input was transposed from samples × taxa to taxa × samples.

Route B used taxonomic names/paths because this pipeline has abundance profiles, not representative microbial sequences required for Route A. Direct name input yielded global fallback for 912/1,068 features. ETE3/NCBI TaxIDs were used to reconstruct taxonomy paths and harmonize current identities; five duplicate mappings were merged, yielding 1,063 taxonomic features with sample abundance totals preserved. Full paths reduced global fallbacks to 18.

Historical MGM2 fallback resolution of the harmonized input:

| Resolution | Count |
| --- | ---: |
| Species | 21 |
| Genus | 686 |
| Family | 269 |
| Order | 41 |
| Class | 28 |
| Global fallback | 18 |
| **Total** | **1,063** |

Downloaded vocabulary was reported to contain 2,223 species prototypes. The early overlap report was 21/822 = 2.55%, using the subsequently corrected rank denominator. Do not present that fraction as the final corrected audit without rechecking the exact eligible species set and unique-identity denominator. The broader finding remains: most GMrepo features fell back to higher taxonomy levels.

MGM2 Route B must be described as a **hierarchical taxonomic representation for this input**, not an embedding preserving every GMrepo species distinctly. Input preparation/resolution files were reported generated; completed downstream MGM2 prediction evaluation is not confirmed. A planned 224-D sample embedding is not a confirmed prediction result.

## 9. Supervised species Transformer V1 — completed historical run

Historical input: 389 × 827 species-ranked TaxIDs, including 138 all-zero rows discovered in v18. The actual saved matrix is in percent units; recovered V1 code divides by 100 after saving it, without strict-subset renormalization. Historical V1 species performance estimates require reassessment in light of this source-resolution failure; no new model estimate has been produced.

For species i in sample s, the token was `t_si = e_i + x_si * a_i`, where both `e_i` (identity) and `a_i` (abundance-direction) are trainable 32-D vectors. All weights started randomly and were learned using ASD labels only. If abundance is zero, the full identity vector `e_i` remains.

| Component | Recorded setting |
| --- | --- |
| Species tokens | 827 |
| Embedding width | 32 |
| CLS token | Yes; total sequence length 828 |
| Encoder | 1 layer, 4 attention heads |
| Feed-forward width | 64 |
| Activation / dropout | GELU / 0.10 |
| Positional encoding | None |
| Sample embedding | Final CLS hidden state, LayerNorm, 32-D |
| Classifier | Linear 32 → 2 logits |
| Training | 20 epochs, batch 16, CPU |
| Optimizer | AdamW; learning rate 0.001; weight decay 0.0001 |
| Loss / gradient clipping | Cross entropy / 1.0 |
| Base seed | 42 |
| Hyperparameter tuning | None |

10-fold results:

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | 0.5217 ± 0.0298 |
| AUC | 0.5228 ± 0.1129 |
| Sensitivity | 0.8000 ± 0.4216 |
| Specificity | 0.2000 ± 0.4216 |
| Precision | 0.4282 ± 0.2258 |
| F1 | 0.5578 ± 0.2941 |

Project-wise results:

| Held-out project | N | Accuracy | AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRJNA1037036 | 123 | 0.4797 | 0.5565 | 1.0 | 0.0 |
| PRJNA451479 | 60 | 0.5000 | 0.5833 | 1.0 | 0.0 |
| PRJNA516054 | 78 | 0.2949 | 0.4747 | 0.0 | 1.0 |
| PRJNA686821 | 128 | 0.5000 | 0.5325 | 1.0 | 0.0 |

Observed conclusion: near-random discrimination and unstable thresholded predictions. The zero-abundance identity background and small abundance contribution are **plausible explanations**, not experimentally established causes. Small labelled sample size, initialization, training dynamics, and other architecture choices may also matter. V1 does not prove Transformers cannot work.

Save V1 unchanged. OOF embeddings generated by independently trained fold models do not share a guaranteed aligned coordinate system. Do not concatenate them and use them as one common feature matrix for another CV model. For downstream embedding evaluation, train/test embeddings must come from the same fold-specific encoder, without supervised-label leakage.

Historical generated filenames:

- `Species_verified_NCBI_matrix.tsv`
- `Species_verified_NCBI_feature_map.tsv`
- `Species_transformer_10fold_metrics.tsv`
- `Species_transformer_10fold_mean_sd.tsv`
- `Species_transformer_10fold_OOF_embeddings.tsv`
- `Species_transformer_projectwise_results.tsv`
- Project-wise embedding files; recover exact names from the actual output directory.

## 10. Organization, scripts, and local environment

Latest recorded analysis root: `/home/ibab/PycharmProjects/Autism_gut_microbiome/`. Some earlier commands used `/home/ibab/PycharmProjects/DL/Autism_gut_microbiome/`. These are historical user-machine paths, not paths present in this workspace. Confirm the current root; do not hard-code inconsistent roots.

Recorded Python interpreter: `/home/ibab/PycharmProjects/DL/nakulan_DL/bin/python` (Python 3.10). V1 used CPU; no CUDA was available in the reported MGM2 setup. Use 10 threads/workers where supported, without changing scientific settings or oversubscribing nested parallelism. Current package versions and available hardware need checking before running new work.

Sources for later October 8 work were corrected to:

- `07_10/GMrepo_ASD_Objective1/1_results/03_species_abundance_long.tsv`
- `07_10/GMrepo_ASD_Objective1/1_results/04_genus_abundance_long.tsv`
- `07_10/GMrepo_ASD_Objective1/1_results/06_sample_metadata_for_modelling.tsv`

Earlier pipeline cleaned script order reported:

1. `01_build_obj1_cohort.py`
2. `02_qc_obj1_cohort.py`
3. `03_freeze_obj1_cohorts.py`
4. `04_prepare_obj1_model_inputs.py`
5. `05_audit_foundation_compatibility.py`
6. `06_audit_mgm_oov_bias.py`
7. `07_decompose_mgm_coverage.py`
8. `08_finalize_mgm_compatibility.py`
9. `09_train_classical_baselines.py`
10. `10_summarize_classical_baselines.py`
11. `11_extract_mgm_frozen_representations.py`
12. `12_compare_mgm_to_classical.py`
13. `13_assess_mgm_robustness.py`

The historical cleanup compiled scripts but explicitly did not rerun the entire pipeline. Do not delete older code/data on the assumption that the cleaned package reproduced all results.

October 8 cleaned package reported `08_10_cleaned/` with:

| Script under `scripts/` | Outputs under `outputs/` |
| --- | --- |
| `00_setup_mgm2_compatibility_patch.py` | MGM2 source compatibility patch |
| `01_baseline_ml.py` | `01_baseline_ml/` |
| `02_gmrepo_curated_probe.py` | `02_curated_audit/` |
| `03_mgm2_prepare_inputs.py` | `03_mgm2/inputs/`, `audits/`, `generated/` |
| `04_mgm2_resolution_audit.py` | MGM2 resolution/taxonomy audits |
| `05_species_transformer_v1.py` | `04_transformer_v1/matrices/`, `metrics/`, and other generated files |

Historical merges: genus/species baseline plus SD reporting → baseline script; MGM2 preparation/path/harmonization → preparation script; initial/corrected taxonomy audit → corrected audit script; taxonomy export → V1 script. The history reports compile checks and unchanged hashes for 31 data/result files; v18 independently verified all 31 compared data/result files are byte-identical across both archives. The cleaned pipeline was not rerun. Actual acceptance/extraction of the cleaned package on the user machine is unconfirmed.

Agreed organization: a dated folder for each research day, scripts together under `scripts/`, experiment-specific numbered output folders under `outputs/`, and a daily `README.md` recording inputs, scripts, outputs, findings, limitations, and next steps. Keep one project-level `PROJECT_STATE.md` as the latest handover. Preserve previous experiment outputs.

For the planned October 9 V2 work, use the same pattern. The earlier suggestion was `scripts/06_species_transformer_v2.py` and `outputs/05_transformer_v2/`; resolve paths within the new dated folder rather than overwriting V1.

## 11. Next actions and pending analyses

1. **Finalize the selected two-project protocol.** The user selected exclusion of the 138 all-zero rows in v20; the derived 251×827 input uses the documented duplicate-corrected candidate. All 827 identities/order are preserved, with 123 ASD and 128 Healthy from PRJNA1037036 and PRJNA686821. Report both excluded projects and the scope change. Do not inherit historical 389-row model results, splits or metadata defaults. Participant independence and full original QC remain unverified here. Recovery of consistent species profiles for all four studies is now an alternative future analysis, not the selected dataset.
2. **Resume V2 only after the matrix/cohort gate is resolved.** Align recovered V1 norm_first=True, zero CLS initialization and project seed=42+100+one-based project number, or declare a reconstructed-control comparison. Start with abundance-gated identity tokens. Keep baseline settings fixed for the initial tokenization ablation. Zero-gated input alone does not guarantee absence has no influence after biased projections/attention; explicitly decide and document masking/pooling behavior, including all-zero samples.
3. **Make the comparison fair.** Preserve V1 settings/splits and run classical ML on the same 827 features. If pooling also changes, add a separate ablation so any improvement is not attributed solely to gating. Use fold-internal validation for any tuning; do not choose V2 using outer test results.
4. **Report both evaluation regimes.** Fold metrics/mean SD plus project holdouts for the explicitly finalized cohort; do not imply four valid species holdouts if only two studies remain. Save predictions, exact splits, config, seed, taxonomy map, model checkpoints where needed, and failure modes. No V2 outcome is currently known.
5. **Return to Objective 1.** Perform representation/domain probes and foundation-model pretraining-overlap audits. Compare frozen, partially fine-tuned, and fully fine-tuned existing models under the same leakage-safe protocol. Waypoint is still a planned comparator, not a completed result.
6. **Audit the proposed large corpus.** Inventory the `53,960` master runs and any existing `GMrepo_53960_full/` files before claiming download completion. Count actually usable profiles; harmonize ranks/TaxIDs; inspect duplicates, subjects, project composition, QC, missingness, and abundance units. Prevent evaluation samples/subjects and, where required by the transfer claim, held-out projects from entering pretraining. Freeze exclusions and vocabulary.
7. **Then pursue Objectives 2 and 3 separately.** Existing pretrained-weight continuation and random-initialized self-supervised pretraining are different experiments. Candidate masked-abundance learning is a proposed task, not an executed model. Evaluate transfer gains, not only training loss; do not call a model a general foundation model solely because it uses a Transformer.
8. **Biological interpretation and reproducibility.** Examine stable taxa/associations across projects and available confounders. Interpretation methods, pathway analysis, and mechanistic claims remain incomplete. Document evidence for any claimed novelty through a fresh literature review when that task is requested.

## 12. Working preferences and boundaries

- Explain science and code clearly, step by step, with simple comments and manageable file counts.
- Keep research-log points human and concise; use tables when exact counts/results are easier to compare.
- Use original data and honest results. Separate completed runs, historical reports, hypotheses, proposals, and illustrative examples.
- Preserve earlier decisions unless a scientific reason for a change is documented.
- The guide-facing boundary previously requested was basic ML: earlier cleaned scripts 01–10. That presentation scope does not mean MGM and Transformer work were undone.
- Do not mix unrelated projects (CRC methylation, temporal chromatin RNNs, or PBMC baseline ZIPs) into this ASD record.
- `baseline08_results.zip` was mentioned in the old export without established ASD contents. Do not import it into this project without identifying it.

## 13. How to maintain this handover

The user explicitly wants a reusable project Markdown file before a chat becomes too long. Keep this file current at major milestones and when the user requests a checkpoint. Do not wait for a guaranteed chat-limit warning: remaining capacity may not be known and a hard limit can prevent an update.

For every update:

1. Advance the checkpoint/date and state which files/logs were actually inspected.
2. Update the top “Read this first” section to reflect the real next action.
3. Add new experiment inputs, exact code/config/split filenames, outputs, genuine results, limitations, and pending items.
4. Preserve earlier results and corrections; label superseded methods instead of erasing them.
5. Keep the same latest `PROJECT_STATE.md` identity when possible, with version history; create dated snapshots only when useful/requested.
6. Save the updated file and give the user a download link.

Daily README contents: date; work completed; input paths; scripts/commands used; output paths; results; errors/fixes; limitations; next day's task. Original archives remain the detailed evidence behind this summary.

### Prompt to start the next chat

> Continue my autism_gut_microbiome research from PROJECT_STATE.md v20. The user selected a separate 251×827 two-project species cohort by excluding all 138 all-zero species rows. Use gmrepo_251_species.zip: 123 ASD, 128 Healthy; retained projects PRJNA1037036 and PRJNA686821; duplicate-corrected percent abundances and unchanged frozen827vocabulary. Preserve the historical389files and documented138exclusions. The oldv18LOCK describes the historical blocked matrix, not the new cohort. No model has run on251; V2 still checks389/fourprojects and needs explicit adaptation plus V1 alignment. Teach one step at a time with examples. Preserve all three objectives; update this handover after verified progress.

## 14. V2 implementation started in this session

The user requested starting V2 after asking about optional plugins. No external integration is needed for local V2 preparation/training. Actual inputs and V1 code were subsequently recovered in v18. They revealed blocking source-resolution/duplicate issues; V2 is paused pending their resolution.

New files: `06_species_transformer_v2.py` and `V2_README.md`. Primary `gated_cls` uses `x*e + x*a`, retains both embedding tables and CLS pooling, and keeps the documented model/training settings. The gated form is algebraically one abundance-scaled effective species vector. `gated_masked_cls` is an explicit separate masking ablation; `additive_cls` is a reconstructed control. V1 details are now independently inspectable: norm_first=True, zero CLS initialization and project seed=42+100+one-based project index. Prepared V2 differs on those details and is not yet a certified gating-only historical comparison.

The script uses required real input paths, explicit units, strict cohort/taxonomy-map checks, exact or generated split manifests, fixed epochs, and separate fold/project outputs. It stops rather than overwriting existing outputs. No tuning or early stopping uses test data. Passed synthetic software checks for complete fold/project runs, metrics, checkpoint loading, alignment, split reuse, units, invalid inputs, gating/masking, and repeated seeded training. Verification used CPU PyTorch 2.14.1+cpu here; these checks are kept separate from real research results. No real-data V2 AUC, prediction, or biological finding exists at this checkpoint.

## 15. Independently verified species audit — v18

User priority: thoroughly verify and lock the 827-species vocabulary from the 1,068 GMrepo features before further V2 work.

**Identity decision:** FROZEN_IDENTITY_VOCABULARY. **Primary matrix/model decision:** BLOCKED_PENDING_COHORT_AND_SOURCE_RESOLUTION. The supplied NCBI snapshot is the rank authority; generic source rank labels are not sufficient.

| Audit fact | Verified result |
| --- | --- |
| Original features | 1,068 |
| Pinned NCBI ranks | 829 species, 237 genus, 1 subspecies, 1 unresolved |
| Unique retained TaxIDs | 827, naturally observed; no target/top-K/prevalence/outcome selection |
| Obsolete-ID resolutions | 11 original features: 7 species, 4 genus |
| Historical species matrix reconstruction | Exact numerical match |
| Current GMrepo source reconstruction | 389/389 profiles; max difference 1.78e-15 percentage points |
| Source feature/TaxID conflicts | 0 |
| Current source project/phenotype agreement | 389/389; QCStatus=1 and Metagenomics for all |
| Outside-species rows labeled species by source | 8,795 |
| Both API profile fields checked | No species-ranked rows hidden in genus; both fields have none for the 138 zero runs |
| All-zero strict-species runs | 138: all 60 PRJNA451479 and all 78 PRJNA516054 |
| Nonzero-species runs | 251: 123 ASD/128 healthy, two studies; not adopted as new primary |
| Species duplicate groups/cells | 46 in 45 runs; 3 TaxIDs |
| Model training in this audit | None |

The 46 cells comprise 42 repeated [Clostridium] methylpentosum source records (TaxID 84026), two Lachnoclostridium sp. An76 alias cases (1965654), and two Pseudoflavonifractor sp. AF19-9AC alias cases (2292244). Retaining one identical source TaxID/value is a documented candidate rule, not a silent overwrite. No different-value collisions occurred. The aliases already shared original TaxIDs; describing the 829→827 reduction exclusively as obsolete-ID merging is inaccurate.

The vocabulary is the pooled original-cohort union. A broad name-pattern check flags 101 retained names containing sp., uncultured or unidentified; describe species-ranked taxonomic nodes rather than 827 independently read-validated formally named species. Handling taxa absent from training studies must be explicit: 18 are exclusive to PRJNA1037036 and 309 to PRJNA686821. Strict inductive vocabulary construction is a separate protocol choice.

Artifacts: SPECIES_827_AUDIT_REPORT.md and species_827_evidence.zip. The ZIP contains the pinned dump, actual inputs, 389 current raw responses, feature crosswalk/exclusions, sample/class coverage, duplicate records, candidate changes, hashes and replay scripts. FROZEN_SPECIES_VOCABULARY.tsv and final LOCK.json are inside the ZIP. The report distinguishes verified results, candidate cleaning and unresolved issues.

Snapshot SHA-256: ea4910c0136193e3d6ea4e40f804caf14449ec56dd64a0fdcebfa5e09eaa3938. Vocabulary SHA-256: bfa319fc5b78b891fae046938b7a0675cfd73a7e696706831d351c4b30e767ac. Main dump member timestamps are 2026-10-08 UTC; original download time is not established. Current source responses were retrieved 2026-10-09 and are not represented as the original acquisition cache.

## 16. Source entries and missing/zero counts — v19

User requested the number of entries per SRR before making the common-column
matrix, and missing/zero counts for every sample and every column of the
389×1068 and 389×827 matrices. Completed independently on 2026-10-09:

- Historical 1068 and 827 matrix bytes match both original `08_10.zip`
  (capitalized filenames) and `08_10_cleaned.zip` (lowercase filenames).
  The candidate hash matches the v18 LOCK.json. All 389 source JSON hashes
  match the v18 source-comparison table. No input files were edited.
- Current cached source responses reconstruct all 415452 original cells,
  max difference 1.7763568394002505e-15 percentage points. The original
  acquisition raw cache is still unavailable; before-filling counts are
  reconstructed from the saved October 9 responses, not an invented recovery.
- Per-sample source records range 3–224, distinct names 3–223, retained
  unique species 0–221. Total source records 35454; distinct-name sample
  positions 35412. Zero source missing-abundance values and zero explicit
  source zeros. The API `species` field includes non-species ranks.
- Original 389×1068: 0 missing NA, 380040 zeros (91.47627162704715%), 35412 positive cells,
  0 all-zero rows, 0 all-zero columns.
- Historical and candidate 389×827 each: 0 missing NA, 295090 zeros (91.7274629083347%),
  26613 positive cells, 138 all-zero rows, 0 all-zero columns. Their
  missing/zero masks are identical; deduplication changes 46 positive values.
- Every matrix zero corresponds to an unreported source entry under the
  applicable feature vocabulary. A zero is not proof of biological absence.
- Example SRR13327451: 159 distinct names and 909 zeros in 1068; 157 unique
  retained species and 670 zeros in 827. SRR10338792: 73 names and 995 zeros
  in 1068; 0 retained species and 827 zeros in 827.
- All per-row and per-column totals were independently reconciled using
  NumPy counts. Historical species merging and candidate source deduplication
  were independently replayed. No model trained, no cohort decision made.
- Per-row and per-column `zero_percent` columns and overall zero percentages
  were added at the user's request. Denominators are 1068/827 features for a
  row, 389 samples for a column, and all matrix positions for the overall rate.
  All percentages independently verified, including range and overall means.

Deliverable: `gmrepo_missingness_check.zip`, containing `check_missing_zeros.py`,
README, 394 checksum-listed input files (three matrices, taxonomy and metadata,
389 JSON responses), eight completed CSV reports, and this handover. The script
uses pandas/numpy, finds files relative to itself, checks all input hashes and
matrix identifiers, verifies all 389 original source-to-matrix rows, and writes
per-sample/per-taxon missing, zero, positive and source-unreported counts plus
zero percentages for every row, every column, and each full matrix.

## 17. Derived two-project species cohort — v20

On 2026-10-09 the user asked to exclude the all-zero rows and use 251×827.
A separate cohort was created from the checked duplicate-corrected candidate;
no historical input or v18 LOCK was overwritten. Exact rule: exclude a sample
only when all 827 retained species abundances are exactly zero.

| Retained project | ASD | Healthy | Total |
|---|---:|---:|---:|
| PRJNA1037036 | 59 | 64 | 123 |
| PRJNA686821 | 64 | 64 | 128 |
| Total | 123 | 128 | 251 |

Excluded PRJNA451479:30ASD+30Healthy=60; PRJNA516054:55ASD+23Healthy=78.
All exclusions and reasons are listed. Excluded profiles lack retrieved
species-ranked entries; this does not mean biological absence of all species.

The new matrix has 207577 abundance cells:26613positive and180964zero,
87.17921542367411%zero. MissingNA=0, all-zero rows=0, all-zero columns=0.
All827species have at least one positive value; frozen column order and every
retained abundance value were verified exactly unchanged after save/reload.
Units remain percent; no renormalization, new taxonomy mapping or prevalence
filter was applied. It inherits the documented candidate duplicate assumption.

Deliverable `gmrepo_251_species.zip`: reproducible `make_251_cohort.py`, README,
original candidate/metadata inputs, unchanged frozen vocabulary, derived matrix
`results/species_251x827_deduplicated_percent.tsv`, matched metadata,138sample
exclusion list, project counts, per-row/per-column zero percentages, overall
summary and SHA256manifest. New matrix SHA256:
`272584a9b335eeaec13e97bd1b4d4c05713e09b20c9f18b7a56b409cf3834e24`.

This is an explicit two-study scope change, not the earlier four-study cohort.
No historical performance estimate applies to it and no new model was trained.
V2's strict389/fourproject checks have not been modified in this step. Adapt
those checks, align V1/control settings, and finalize same-input comparisons
and evaluation before training. Keep the user's explanation to one step at a time.

## 18. Checkpoint log

| Checkpoint | Date | Content |
| --- | --- | --- |
| v16, historical filename | 2026-10-07 | Frozen MGM versus classical ML/robustness; original file located, not reread for this audit |
| v17, this reconstruction | 2026-10-09 | Restores essential historical state, exact three objectives, later ML/MGM2/taxonomy/V1 work, V2 code preparation, pending real-data V2/pretraining, and continuation instructions |
| v18, independent audit | 2026-10-09 | Archives recovered; all 389 current source profiles replayed; 827 identities frozen; 138 all-zero runs and 46 duplicate abundance cells identified; primary species modeling paused |
| v19, missing/zero counts | 2026-10-09 | Exact files checked against both archives; source entry counts and all matrix row/column NA/zero counts verified; simple reproducible code and full inputs/results packaged; primary species modeling remains paused |
| v20, derived251cohort | 2026-10-09 | User-selected exclusion of138all-zero species rows; separate251×827 two-study dataset created from deduplicated candidate; 123ASD/128Healthy, all827columns retained, exact values preserved, exclusions documented; no model trained |

**No new ASD model training or causal biological finding occurred. New source-response retrieval and independent taxonomy/matrix audits were completed.**
