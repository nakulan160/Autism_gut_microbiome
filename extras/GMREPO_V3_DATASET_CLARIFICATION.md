# GMrepo dataset confusion: checked facts and current decision

Prepared October 11, 2026. Source-response audit: October 10, 2026. No model training was performed for this explanation.

**Current scope:** use species profiles already available through GMrepo v 3. The reviewed bulk-only input is **128 runs × 809 microbial species, 64 ASD / 64 Healthy**, from PRJNA686821. It is a preliminary single-project dataset, pending participant and clinical eligibility checks. The old 251×827 matrix is preserved as historical evidence and is superseded for the ordinary bulk-species endpoint. AGORA, viral table recovery, raw-read reprofiling and pooling v 2 abundance files are outside the current user-selected scope.

## 1. Three sources were being confused

| Source | What it supplies or analyses | Relation to our work |
|---|---|---|
| GMrepo v 3 | Current run metadata and processed microbial relative-abundance profiles | Current data source |
| Babolin et al., BMC Psychiatry, 2026 | A different GMrepo v 2 collection used for theoretical bacterial metabolite-production analysis | Methodological/contextual reference; not our cohort and not a viral abundance study |
| Wan et al., Cell Reports Medicine, 2024 | Virus-enriched stool sequencing; viral analysis computed from reads; bacterial analysis using earlier bulk sequencing | Establishes the preparation of PRJNA1037036 and explains its relationship to earlier bulk data |

The BMC paper's combination is not bacterial-plus-viral combination. Its reported 331 ASD and 5,150 controls are not the count of our current bulk cohort. AGORA details remain deferred.

## 2. What was checked correctly, and what the initial screening missed

| Verification | Result |
|---|---|
| Original 389×1068 abundance reconstruction | All 415,452 cells compared against fresh source responses; maximum difference 1.7763568394002505e-15, below 1e-12 tolerance |
| Processed 251×827 abundance reconstruction | All 207,577 cells compared against fresh source responses; exact agreement |
| 827 retained species identities | Checked against the preserved NCBI taxonomy snapshot; valid species-ranked TaxIDs |
| Source candidate responses |464 run responses retrieved across the five identified projects |
| Experimental preparation and related libraries | Later audit identified VLP enrichment and matching bulk metadata; the initial screening had not adequately recognized these |

Accurate reconstruction does not establish a complete eligible cohort, independent participants, compatible experimental preparations, complete metadata or identical per-run profiling versions. The processed matrix uses documented rank filtering, TaxID grouping and duplicate handling; it is not an unchanged copy of every source feature.

## 3. Why the dimensions changed

| Stage | Dimensions/count | What happened |
|---|---|---|
| Identified candidate inventory |464 runs |340 bulk candidates plus 124 VLP-enriched candidates |
| Original retrieved modeling matrix |389×1068;208 ASD/181 Healthy |74 empty-profile candidates and 1 QC-failed VLP run were left out |
| Strict species vocabulary |829 original species-ranked features → 827 unique current TaxIDs | The 1,068 dictionary contained mixed taxonomic ranks; current-TaxID merging reduced 829 to 827 |
| Strict species matrix before dropping empty rows |389×827 |138 older bulk runs had no retained species-level entries in the retrieved profiles |
| Old selected matrix |251×827;123 ASD/128 Healthy |138 all-zero retained-species rows were excluded |
| Bulk-only rows |128×827;64 ASD / 64 Healthy |123 VLP-enriched rows excluded from the ordinary bulk endpoint |
| Reviewed bulk-only matrix |128×809 |18 columns had no positive values after excluding VLP rows and were removed |

The 138 all-zero rows were not evidence that those stools lacked species. Their retrieved records lacked usable species resolution. The 18 removed columns were zero in this bulk subset; that does not establish universal biological absence.

The 827 historical dictionary contains 824 bacteria, 2 eukaryotes and 1 archaeon. The 809 bulk dictionary contains 806 bacteria, 2 eukaryotes and 1 archaeon. Therefore call the selected endpoint **microbial species** unless a separate bacterial-only domain filter is explicitly chosen.827 is a historical feature union, not a universal required number of species.

Two different duplication issues must not be confused: equal-abundance name variants sharing an original TaxID were counted once under the documented duplicate rule; distinct source TaxIDs mapped to one current TaxID were grouped. Separately, bulk/VLP metadata matches concern related experimental libraries, not duplicated taxon columns. Historical inputs and processing decisions are preserved.

## 4. What the 251 rows actually represent

| Project | Preparation | ASD | Healthy | Retained runs | Decision for current endpoint |
|---|---|---:|---:|---:|---|
| PRJNA686821 | Bulk stool shotgun |64|64|128|Retain for preliminary review |
| PRJNA1037036 | Virus-like-particle-enriched shotgun |59|64|123|Exclude from ordinary bulk-species modeling |
| Old total | Both preparations mixed |123|128|251|Not an independent bulk cohort |

WGS/METAGENOMIC metadata alone do not prove that a library is an ordinary bulk preparation. Virus-enriched libraries also use shotgun sequencing.

All 123 retained VLP libraries match a bulk library by normalized sample alias, collection date and phenotype. Example: PWHRCA001/SRR26718184(VLP) matches a 001/SRR13327535(bulk), November 3, 2017, ASD. The normalization removes the exact leading PWHRC and lowercases. Their FASTQ checksums differ; these are different assays, not byte-identical sequencing files. An author participant-identity crosswalk is unavailable, so metadata matches are not presented as certified unique-person identities.

Related assays must not be treated as independent people or split across training/testing as though unrelated. Grouping related libraries could address that evaluation concern, but it would not make their bacterial measurements experimentally comparable.

The full VLP project has 124 candidates(60 ASD/64 Healthy). The extra ASD run SRR26718155 has GMrepoQCStatus 0 and was absent from the old 251 matrix. It matches bulk SRR13327500/a 019. Its bacterial-profile QC failure does not independently establish viral unusability. It remains a historical audit candidate and is not added to the current bulk matrix.

## 5. Does GMrepo provide bacterial abundances for VLP libraries?

**Yes, in the checked responses.** For VLP run SRR26718184, the species response contains:

| Bacterial taxon | Reported relative abundance |
|---|---:|
| Segatella copri |66.96061%|
| Phocaeicola vulgatus |15.21510%|
| Bacteroides xylanisolvens |3.47792%|

These values are actual saved source values. Their availability does not establish that they measure the original bulk stool bacterial community. The source response alone does not provide complete per-run command logs or establish whether every individual assignment is biologically correct.

The researchers enriched viruses to improve detection and assembly of low-abundance viruses. They did not choose viral enrichment as the preparation for their bacterial-community analysis. Residual bacterial DNA is a plausible reason for bacterial signal after enrichment; classification errors could also contribute. The specific cause for each GMrepo assignment has not been established. Do not claim proven contamination, database error or equivalent bulk measurements.

## 6. Where the original study's bacterial and viral values came from

| Measurement in Wan et al. | Source and processing |
|---|---|
| Bacterial composition | Raw bulk shotgun metagenomes from the authors' earlier study; trimming/human-read removal; MetaPhlAn 2 profiling |
| Viral composition | Virus-enriched sequencing; read cleaning; MEGAHIT assembly; viral-contig selection/catalog; BWA mapping; SAMtools abundance table; TPM normalization |

The paper explicitly separates these pipelines. Its bacterial results should not be equated with GMrepo's bacterial entries for VLP libraries. The authors could calculate viral abundances without GMrepo supplying those abundances: they had their own reads, catalog and analysis pipeline.

The inspected article and Document S1 supply viral results and group statistics, but no complete downloadable sample-by-virus abundance matrix was found there. Deposited FASTQ files are raw sequences, not processed abundance values. No missing viral measurements were inferred, zero-filled or generated in our work. Reconstructing viral values or contacting authors is outside the current user-selected scope.

## 7. Why viral features are absent from our export

| Audit target | Finding |
|---|---|
| Original 1,068 feature dictionary |1061 bacterial,4 eukaryotic,2 archaeal,1 Unknown;zero viral features |
| All 124 current VLP species responses |518 distinct known TaxIDs, all bacterial; Unknown entries also present;zero viral TaxIDs |
| Historical 827 retained dictionary |Zero viral species |

The viral features were absent **before** the 827-species filter. Unknown (−1) cannot be treated as a viral measurement. Absence from the export does not mean absence from raw reads.

GMrepo v3 describes default MetaPhlAn 4 v4.1.0 profiling and identifies viral abundance as a future addition. Official MetaPhlAn 4.1 documentation describes a separate optional viral module(--profile_vsc/--vsc_out), whose output is viral sequence-group coverage. That supports the export explanation; no exact command is asserted for an individual GMrepo run. It does not establish that v2 provided complete virome tables subsequently removed from v3.

## 8. Why normalized percentages do not solve compatibility

GMrepo's methods describe relative abundances normalized to 100% at genus/species levels. Percentages are not absolute cell counts and are not automatically percentages of all raw sequencing reads.

**Illustrative numbers only:**

| Stage | Bacterium A | Bacterium B |
|---|---:|---:|
| Original material |10 units|90 units|
| A preparation removes most B |10 units|1 unit|
| Percentages after that preparation |90.9%|9.1%|

Normalization cannot restore material removed by preparation. Bulk projects can also differ in extraction, sequencing and profiling; identical percentage units alone do not prove comparability.

Our selected matrix preserves the existing percentages without renormalizing after taxonomic filtering. Selected-species row sums may therefore be below 100% because Unknown or excluded taxa are omitted. Missing NA, zero for an unreported feature within an available profile, and an unavailable entire profile are different conditions; do not substitute one for another.

## 9. What happened to older projects and the Autistic Disorder label

| Older bulk project | Candidate runs | Checked current profile limitation |
|---|---:|---|
| PRJNA451479 |60|Genus-ranked entries in the legacy species field; tested newer endpoints returned empty profiles |
| PRJNA516054 |78|Same resolution/availability limitation |
| PRJEB23052 |74|Empty taxonomic profile fields |
| Total |212|Not available as a usable current species matrix through checked routes |

These projects remain listed in v 3. Their internal migration/processing/API cause is undocumented in the checked materials. We cannot say the projects or all their raw data were deleted. Species-ranked entries exist in older author exports for the first two projects, but old exports are not automatically interchangeable with current profiles.

The separate older Autistic Disorder label is not currently returned as a separate phenotype. All 163 runIDs in the old Autistic Disorder genus file were found under current ASD; all 89 species-file IDs were among them and were currently amplicon/QC-failed. This supports consolidation of those checked records under ASD, not disappearance of all their data.

Separately, 53 runIDs from the older ASD-labeled collection were absent from the current ASD listing. Their status/cause remains unconfirmed; absent from that listing is not proof of deletion from all sources or a Healthy phenotype.

The v 2/v 3 publications describe different profiling software/databases. They do not prove every old run was reprocessed identically with a newer pipeline. The live v 2 service was unavailable during our audit; historical comparisons used pinned author files and primary papers.

## 10. Why the BMC paper's counts are much larger

The paper reports 331 ASD/5150 controls from GMrepo v 2. Our audit of its ASD files found 203 amplicon and 128 shotgun runs. Its shotgun ASD 128 consists of 43 PRJEB23052, 30 PRJNA451479 and 55 PRJNA516054. Our current bulk 128 consists of 64 ASD and 64 Healthy from PRJNA686821. **These two 128 counts describe different collections.** There is zero run-accession overlap between the old 251 selected collection and the author's inspected species files.

The authors used a much larger Health control reservoir, rather than only controls from our selected projects. A species filename or large taxon-record count does not prove shotgun sequencing or an independent participant count. There are repeated BioSamples among author control runs. Their larger cohort does not by itself establish a retrieval failure in our selected runs.

The public control files have 5,152 raw runs; the earlier reconciliation identified exactly 2 Unknown-only runs omitted by the authors' processing, yielding 5,150. This count discrepancy is resolved and is separate from assay compatibility. Metabolic-model details remain deferred.

The checked four-project bulk candidate inventory is 340 runs(192 ASD / 148 Healthy), not a completed 340-person cohort. Under the current v 3-available-profile-only scope, the other 212 cannot be added as fabricated zeros or pooled from unharmonizedv 2 abundances. This inventory is not claimed to cover every eligible bulk project in all of GMrepo v 3.

## 11. Current decision and limits

| Item | Decision |
|---|---|
| Data acquisition scope |Already available GMrepo v 3 profiles |
| Preliminary species input | PRJNA686821;128 × 809; 64 ASD / 64 Healthy |
| VLP libraries |Preserve as provenance;exclude from ordinary bulk-species model |
| Unavailable older profiles |Document;exclude from this available-profile analysis;do not zero-fill |
| Viral analysis/raw-read reconstruction/v 2 pooling/AGORA |Deferred under user instruction |
| Before locking/modeling |Review available participant identity and clinical eligibility; document any unresolved metadata |
| Claims supported now |Source reconstruction, taxonomy and assay audit |
| Claims not established |Final independent-participant cohort, publication readiness, multicohort generalization, viral measurements or new model performance |

The current 128 × 809 matrix is a reviewed processed subset from current responses, not the whole GMrepo v 3 cohort and not proof of complete per-run pipeline uniformity. Clinical/sample metadata limitations may prevent stronger claims. Earlier model metrics on other cohorts cannot be transferred to this newly selected dataset. A supervised classifier on 128 runs alone is not evidence of successful large-scale foundation-model pretraining.

**Guide-facing conclusion:** Our original values were reproduced correctly, but the old 251-run cohort mixed bulk and virus-enriched libraries with matching sample metadata. We are restricting the present analysis to the available 128 bulk runs and 809 microbial species from GMrepo v 3, with 64 ASD and 64 Healthy. Older unavailable profiles, viral analyses and v 2 pooling are outside this scope. Participant/clinical eligibility remains to be checked before a final lock.

## References and audit evidence

- [GMrepo v3 paper](https://academic.oup.com/nar/article/54/D1/D734/8340991).
- [GMrepo v2 paper](https://academic.oup.com/nar/article/50/D1/D777/6426060).
- [Babolin et al., BMC Psychiatry,2026](https://link.springer.com/article/10.1186/s12888-026-08178-8).
- [Wan et al., Cell Reports Medicine,2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897546/).
- [MetaPhlAn 4.1 official documentation](https://github.com/biobakery/biobakery/wiki/MetaPhlAn-4.1).
- Main audit artifacts: DATASET_REALITY_REPORT.md, DATASET_AUDIT_EVIDENCE.zip, SHOTGUN_VIROME_PAIRING.zip and PROJECT_STATE.md.
- Exact proofs: current_source_reconstruction_checks.json,27_original_1068_domains_checked.tsv,viral_source_trace_summary.json,11_phenotype_run_reconciliation.tsv,18_VLP_bulk_alias_date_phenotype_matches.tsv and candidate_pairing_124.tsv.

Author-paper method claims and our independent audit findings are different evidence types. The primary papers do not certify our reconstruction counts or metadata-pairing results; those are reproduced in the saved audit files.
