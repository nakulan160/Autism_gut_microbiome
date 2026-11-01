#!/usr/bin/env python3
"""Prepare five linked GMrepo tables using all eligible QC-passed phenotypes.

Python 3.10+, standard library only; no network, API key or GPU required.
Run: python3 prepare_metagenomic_X.py --source metagenomic_totla --workers 10
Default output: SOURCE/prepared_metagenomic (five tables and one summary).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import contextlib
import csv
import datetime as dt
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import itertools
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import unicodedata

VERSION = "2.0.0"
LOG = logging.getLogger("prepare_X")
RUN = re.compile(r"(?:SRR|ERR|DRR)[0-9]+\Z")
MESH = re.compile(r"\b[DC][0-9]{6,9}\b")
ASD_CODES = {"D000067877", "D001321", "D020817"}
ASD_WORDS = re.compile(r"\b(?:autism|autistic|asd|asperger(?:s)?)\b", re.I)
AMBIGUOUS_ASD = re.compile(r"\b(?:non|no|not|without|negative|control|controls|sibling|siblings)\b", re.I)
SOURCE_FILES = ("all_metagenomic_metadata.tsv", "all_species_abundance_long.tsv.gz",
                "taxa_dictionary.tsv", "extraction_manifest.tsv")
DATA_FILES = ("all_metadata.tsv.gz", "qc_pass_abundance.tsv.gz", "qc_fail_abundance_long.tsv.gz",
              "asd_qc_pass_abundance.tsv.gz", "non_asd_qc_pass_abundance.tsv.gz")
RESULT_FILES = (*DATA_FILES, "filtering_summary.json")
UNAVAILABLE = {"no_species_field", "empty_species_profile", "not_found"}
GENERIC_NAMES = {"unknown", "unclassified", "unidentified", "other", "others"}
AUDIT_COLUMNS = ("prep_qc_decision", "prep_qc_values_json", "prep_asd_group", "prep_asd_evidence_json",
                 "prep_sequencing_decision", "prep_disposition", "prep_exclusion_reasons_json",
                 "prep_in_qc_pass_matrix", "prep_in_asd_matrix", "prep_in_non_asd_matrix",
                 "prep_retained_features", "prep_retained_abundance_sum", "prep_discarded_abundance_sum",
                 "prep_duplicate_records_collapsed", "prep_duplicate_review_json",
                 "prep_project_ids_json", "prep_sample_ids_json", "prep_subject_ids_json",
                 "prep_source_raw_file", "prep_source_sha256")
AVAILABLE = {"downloaded", "downloaded_with_flags"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jd(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": Path(path).stat().st_size}


def atomic_json(path, data, compressed=False):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            raw = (jd(data) + "\n").encode()
            f.write(gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def check(condition, message):
    if not condition:
        raise ValueError(message)


@contextlib.contextmanager
def gzip_text(path, level=4):
    # Stable gzip headers make unchanged output data byte-for-byte reproducible.
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=level, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                yield text


def natural_taxid(value):
    s = str(value).strip()
    return int(s) if re.fullmatch(r"[1-9][0-9]*", s) else None


def abundance(value):
    try:
        n = Decimal(str(value))
        return n if n.is_finite() and n >= 0 else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def list_cell(value, field):
    check(value not in (None, ""), "Missing JSON array in " + field)
    result = json.loads(value)
    check(isinstance(result, list), field + " must contain a JSON array")
    return result


def scalar_values(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from scalar_values(item)
    elif value is not None and str(value).strip():
        text = str(value).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                pass
            else:
                if isinstance(parsed, list):
                    yield from scalar_values(parsed)
                    return
        yield text


def normalized_text(text):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def qc_result(values):
    canonical = set()
    for value in values:
        try:
            n = Decimal(value)
        except InvalidOperation:
            canonical.add(value)
        else:
            canonical.add(str(int(n)) if n.is_finite() and n == int(n) else value)
    if not canonical:
        return "missing", []
    if canonical == {"1"}:
        return "pass", ["1"]
    if canonical == {"0"}:
        return "fail", ["0"]
    return ("conflict" if {"0", "1"} <= canonical else "unrecognized"), sorted(canonical)


def read_tsv(path, required):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        check(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
              "Missing or duplicate columns: " + str(path))
        check(set(required) <= set(reader.fieldnames), "Required columns missing: " + str(path))
        for row in reader:
            check(None not in row and all(v is not None for v in row.values()), "Malformed TSV row: " + str(path))
            yield row


def inspect_metadata(row, manifest):
    rid = row["run_id"]
    check(RUN.fullmatch(rid), "Invalid metadata run ID: " + rid)
    observed = json.loads(row["all_observed_metadata_json"])
    check(isinstance(observed, dict), "Observed metadata must be a JSON object: " + rid)
    qc_values = set(scalar_values(list_cell(row["qc_status_values"], "qc_status_values")))
    mesh = set(scalar_values(list_cell(row["phenotype_mesh_ids"], "phenotype_mesh_ids")))
    terms = set(scalar_values(list_cell(row["phenotype_names"], "phenotype_names")))
    seq, projects, samples, subjects = set(), set(), set(), set()
    for key, values in observed.items():
        key_lower = key.casefold()
        vals = set(scalar_values(values))
        if key_lower == "qcstatus":
            qc_values.update(vals)
        if key_lower in {"experiment_type", "library_strategy"}:
            seq.update(vals)
        if key_lower in {"disease", "mesh_id", "diseases", "_associated_phenotype_mesh_id"}:
            for v in vals:
                mesh.update(MESH.findall(v))
        if key_lower in {"phenotype", "phenotypes", "term", "_associated_phenotype_name"}:
            terms.update(vals)
        if key_lower in {"project_id", "bioproject", "bioproject_id", "study_accession"}:
            projects.update(vals)
        if key_lower in {"sample_id", "biosample", "biosample_id", "biosample_accession", "sample_accession"}:
            samples.update(vals)
        if key_lower in {"subject_id", "participant_id", "host_subject_id", "individual_id", "host_id"}:
            subjects.update(vals)
    # The original columns are another preserved observation, not an override.
    if row.get("QCStatus"):
        qc_values.update(scalar_values(row["QCStatus"]))
    if row.get("experiment_type"):
        seq.update(scalar_values(row["experiment_type"]))
    qc, qc_vals = qc_result(qc_values)
    accepted_types = {"metagenomics", "metagenomic", "shotgun metagenomics", "metagenomic shotgun",
                      "whole metagenome shotgun", "wgs"}
    seq_norm = {normalized_text(v) for v in seq}
    sequencing = "confirmed_metagenomic" if seq_norm and seq_norm <= accepted_types else "metagenomic_inventory_only" if not seq_norm else "conflicting_sequencing_annotation"
    evidence = ["MeSH:" + m for m in sorted(mesh & ASD_CODES)]
    if row.get("asd_label_observed", "").casefold() == "yes":
        evidence.append("extractor_ASD_flag")
    ambiguous = []
    for term in sorted(terms):
        norm = normalized_text(term)
        if ASD_WORDS.search(norm):
            if AMBIGUOUS_ASD.search(norm):
                ambiguous.append(term)
            else:
                evidence.append("phenotype:" + term)
    asd = "ambiguous_ASD_annotation" if ambiguous else "ASD_label_observed" if evidence else "no_ASD_label_observed"
    reasons = []
    if qc != "pass":
        reasons.append("qc_" + qc)
    if sequencing == "conflicting_sequencing_annotation":
        reasons.append(sequencing)
    if manifest["status"] not in AVAILABLE:
        reasons.append("profile_" + manifest["status"])
    check(row["profile_status"] == manifest["status"], "Metadata and manifest profile status differ: " + rid)
    return {"run_id": rid, "qc": qc, "qc_values": qc_vals, "sequencing": sequencing, "asd": asd,
            "asd_evidence": evidence + ["ambiguous:" + t for t in ambiguous],
            "projects": sorted(projects), "samples": sorted(samples), "subjects": sorted(subjects),
            "reasons": reasons, "species": 0, "retained_sum": "", "discarded_sum": "",
            "collapsed": 0, "duplicate_review": [], "retained_records": 0}



def raw_duplicate_proof(source, manifest, rows):
    path = (source / manifest["source_raw_file"]).resolve()
    check(path.is_relative_to(source), "Raw response path leaves the extraction folder")
    if not path.is_file():
        return False
    with gzip.open(path, "rb") as f:
        raw = f.read()
    check(sha(raw) == manifest["source_sha256"], "Raw profile checksum failed: " + manifest["run_id"])
    obj = json.loads(raw, parse_float=Decimal)
    check(isinstance(obj, dict) and isinstance(obj.get("species"), list), "Raw profile has no species array")
    species = obj["species"]
    check(len(species) == int(manifest["records_returned"]), "Raw species count differs from manifest")
    signatures = []
    names = set()
    for row in rows:
        rec = species[int(row["record_index"]) - 1]
        check(isinstance(rec, dict), "Duplicate raw record is not an object")
        tid = str(rec.get("ncbi_taxon_id", rec.get("taxon_id", "")))
        name = str(rec.get("scientific_name", rec.get("name", "")))
        value = rec.get("relative_abundance", rec.get("abundance"))
        check(tid == row["ncbi_taxon_id"] and name == row["scientific_name"]
              and abundance(value) == abundance(row["relative_abundance"]), "Raw and exported duplicate records differ")
        names.add(re.sub(r"[^a-z0-9]", "", normalized_text(name)))
        signatures.append(jd({k: v for k, v in rec.items()
                              if k not in {"scientific_name", "name", "relative_abundance", "abundance"}}))
    return len(names) == 1 and len(set(signatures)) == 1



def fingerprints(folder, names, workers):
    with futures.ThreadPoolExecutor(max_workers=min(workers, len(names))) as pool:
        return dict(zip(names, pool.map(lambda name: fingerprint(folder / name), names)))


def catalogue_key(row):
    return (row["ncbi_taxon_id"], row["scientific_name"], row["reported_rank"])


def local_catalogue(catalogue):
    decisions, names = {}, collections.defaultdict(collections.Counter)
    counts, reasons = {}, collections.Counter()
    for row in catalogue:
        key = catalogue_key(row)
        check(key not in counts, "Duplicate catalogue key")
        count = int(row["records"])
        check(count > 0, "Nonpositive catalogue record count")
        counts[key] = count
        tid = natural_taxid(key[0])
        name = re.sub(r"\s+", " ", key[1]).strip()
        rank = key[2].strip().casefold()
        reason = ("invalid_or_placeholder_id" if tid is None else "missing_name" if not name
                  else "unknown_name" if normalized_text(name) in GENERIC_NAMES
                  else "reported_rank_not_species" if rank and rank != "species"
                  else "keep_provider_species_entry")
        decisions[key] = tid if reason == "keep_provider_species_entry" else None
        reasons[reason] += 1
        if decisions[key] is not None:
            names[tid][name] += count
    registry = {}
    for tid, variants in names.items():
        # Use an actual supplied name, never an externally resolved name.
        name = min(variants, key=lambda n: (-variants[n], -sum(c in ".-" for c in n), n.casefold(), n))
        registry[tid] = {"original_taxid": tid, "representative_name": name,
                         "original_names": sorted(variants),
                         "name_record_counts": dict(sorted(variants.items()))}
    return decisions, registry, counts, reasons


def feature_headers(feature_ids, registry):
    counts = collections.Counter(registry[t]["representative_name"].casefold() for t in feature_ids)
    used, headers = {"run_id"}, {}
    for tid in sorted(feature_ids, key=lambda t: (registry[t]["representative_name"].casefold(), t)):
        name = registry[tid]["representative_name"]
        header = name if counts[name.casefold()] == 1 and name.casefold() != "run_id" else name + "__taxid_" + str(tid)
        while header.casefold() in used:
            header += "__taxid_" + str(tid)
        used.add(header.casefold())
        headers[tid] = header
    return headers


def clean_run(rows, state, manifest, decisions, source):
    groups, discarded = collections.defaultdict(list), Decimal(0)
    for row in rows:
        n = abundance(row["relative_abundance"])
        if n is None:
            state["reasons"].append("invalid_abundance")
            continue
        tid = decisions[catalogue_key(row)]
        if tid is None:
            discarded += n
        else:
            groups[tid].append(row)
    values = {}
    for tid, records in groups.items():
        if len(records) > 1:
            equal = len({abundance(r["relative_abundance"]) for r in records}) == 1
            proven = equal and raw_duplicate_proof(source, manifest, records)
            state["duplicate_review"].append({
                "original_taxid": tid, "record_indices": [int(r["record_index"]) for r in records],
                "values": [r["relative_abundance"] for r in records],
                "action": "collapse_proven_repeat" if proven else "hold_run_for_review"})
            if not proven:
                state["reasons"].append("conflicting_or_unproven_duplicate_measurements")
                continue
            state["collapsed"] += len(records) - 1
            discarded += sum((abundance(r["relative_abundance"]) for r in records[1:]), Decimal(0))
        values[tid] = records[0]["relative_abundance"]
    retained = sum((abundance(v) for v in values.values()), Decimal(0))
    state.update(species=len(values), retained_sum=str(retained), discarded_sum=str(discarded))
    if retained <= 0:
        state["reasons"].append("no_positive_named_features_remaining")
    state["reasons"] = sorted(set(state["reasons"]))
    return values


def metadata_additions(state, manifest):
    included = not state["reasons"]
    in_asd = included and state["asd"] == "ASD_label_observed"
    in_non_asd = included and state["asd"] == "no_ASD_label_observed"
    values = [state["qc"], jd(state["qc_values"]), state["asd"], jd(state["asd_evidence"]), state["sequencing"],
              "included_in_qc_pass_matrix" if included else "excluded_from_qc_pass_matrix",
              jd(sorted(set(state["reasons"]))), int(included), int(in_asd), int(in_non_asd),
              state["species"], state["retained_sum"], state["discarded_sum"], state["collapsed"],
              jd(state["duplicate_review"]), jd(state["projects"]), jd(state["samples"]), jd(state["subjects"]),
              manifest["source_raw_file"], manifest["source_sha256"]]
    return dict(zip(AUDIT_COLUMNS, values))


def validate_matrices(work, headers, expected, feature_ids):
    LOG.info("Checking all three matrices and their exact subset rows")
    positive_columns = set()
    with contextlib.ExitStack() as stack:
        readers = [csv.reader(stack.enter_context(gzip.open(work / name, "rt", newline="", encoding="utf-8")), delimiter="\t")
                   for name in (DATA_FILES[1], DATA_FILES[3], DATA_FILES[4])]
        for reader in readers:
            check(next(reader, None) == headers, "Matrix headers differ")
        counts = collections.Counter()
        for row, item in itertools.zip_longest(readers[0], expected):
            check(row is not None and item is not None, "Matrix row count differs")
            rid, asd = item
            check(row[0] == rid and len(row) == len(headers), "Matrix row identity/width differs")
            counts["qc_pass"] += 1
            positive = False
            for index, value in enumerate(row[1:]):
                if value != "0":
                    n = abundance(value)
                    check(n is not None, "Invalid matrix abundance")
                    if n > 0:
                        positive = True
                        positive_columns.add(index)
            check(positive, "Empty matrix profile")
            if asd == "ASD_label_observed":
                check(next(readers[1], None) == row, "ASD subset differs from main matrix")
                counts["ASD"] += 1
            elif asd == "no_ASD_label_observed":
                check(next(readers[2], None) == row, "Non-ASD subset differs from main matrix")
                counts["no_ASD_annotation"] += 1
            else:
                counts["ambiguous_ASD"] += 1
        check(next(readers[1], None) is None and next(readers[2], None) is None, "Extra subset rows")
    check(len(positive_columns) == len(feature_ids), "All-zero feature in main matrix")
    return counts


def build(source, out, source_hashes, summary, args, parameters, original_summary_bytes):
    LOG.info("Stage 2/5: classify QC and phenotype annotations; read the local feature catalogue")
    manifest = {}
    for row in read_tsv(source / SOURCE_FILES[3], {"run_id", "status", "records_returned", "records_exported",
                                                  "source_raw_file", "source_sha256", "error"}):
        rid = row["run_id"]
        check(RUN.fullmatch(rid) and rid not in manifest, "Invalid/repeated manifest run: " + rid)
        check(row["status"] in AVAILABLE | UNAVAILABLE and not row["error"], "Unresolved profile request: " + rid)
        check(re.fullmatch(r"[a-f0-9]{64}", row["source_sha256"]) and row["source_raw_file"], "Missing raw-profile provenance: " + rid)
        check(int(row["records_exported"]) == int(row["records_returned"]) >= 0, "Profile record counts differ: " + rid)
        check(row["status"] in AVAILABLE or int(row["records_exported"]) == 0, "Unavailable profile has records: " + rid)
        manifest[rid] = row
    check(len(manifest) == summary["counts"]["runs"], "Manifest run total differs from extraction summary")
    catalogue = list(read_tsv(source / SOURCE_FILES[2], {"ncbi_taxon_id", "scientific_name", "reported_rank", "records"}))
    decisions, registry, catalogue_counts, catalogue_decisions = local_catalogue(catalogue)
    states, qc_counts, asd_counts = {}, collections.Counter(), collections.Counter()
    for index, row in enumerate(read_tsv(source / SOURCE_FILES[0], {"run_id", "qc_status_values", "phenotype_mesh_ids",
                    "phenotype_names", "all_observed_metadata_json", "profile_status"}), 1):
        rid = row["run_id"]
        check(rid in manifest and rid not in states, "Metadata inventory differs: " + rid)
        state = inspect_metadata(row, manifest[rid])
        states[rid] = state
        qc_counts[state["qc"]] += 1
        asd_counts[state["asd"]] += 1
        if index % 10000 == 0:
            LOG.info("Metadata checked: %s", f"{index:,}")
    check(states.keys() == manifest.keys(), "Metadata/manifest inventories differ")
    LOG.info("QC decisions: %s; ASD annotations do not exclude runs from the main matrix", dict(qc_counts))
    with tempfile.TemporaryDirectory(prefix="building-", dir=out) as directory:
        work = Path(directory)
        intermediate = work / "retained_rows.jsonl.gz"
        observed, processed, feature_ids, kept_ids = collections.Counter(), set(), set(), []
        failed_records, failed_profile_ids = 0, set()
        LOG.info("Stage 3/5: stream original abundances and preserve QC-failed records")
        long_rows = read_tsv(source / SOURCE_FILES[1], {"run_id", "record_index", "ncbi_taxon_id", "scientific_name",
                                    "reported_rank", "relative_abundance", "source_raw_file", "source_sha256"})
        with gzip.open(source / SOURCE_FILES[1], "rt", newline="", encoding="utf-8-sig") as f:
            long_header = next(csv.reader(f, delimiter="\t"))
        last = ""
        with gzip_text(intermediate, args.compression_level) as kept, gzip_text(work / DATA_FILES[2], args.compression_level) as failed:
            failed_writer = csv.DictWriter(failed, fieldnames=long_header, delimiter="\t", lineterminator="\n")
            failed_writer.writeheader()
            for group_index, (rid, group) in enumerate(itertools.groupby(long_rows, key=lambda r: r["run_id"]), 1):
                check(rid in states and rid > last, "Abundance runs are unknown, repeated or unsorted: " + rid)
                last = rid
                state, m = states[rid], manifest[rid]
                check(m["status"] in AVAILABLE, "Records for unavailable profile: " + rid)
                records = []
                for index, row in enumerate(group, 1):
                    check(index <= int(m["records_exported"]) and int(row["record_index"]) == index, "Invalid abundance record sequence: " + rid)
                    check(row["source_raw_file"] == m["source_raw_file"] and row["source_sha256"] == m["source_sha256"], "Abundance provenance differs: " + rid)
                    key = catalogue_key(row)
                    check(key in catalogue_counts, "Abundance taxon absent from catalogue: " + rid)
                    observed[key] += 1
                    records.append(row)
                    if state["qc"] == "fail":
                        failed_writer.writerow(row)
                        failed_records += 1
                        failed_profile_ids.add(rid)
                check(len(records) == int(m["records_exported"]), "Missing abundance records: " + rid)
                processed.add(rid)
                if not state["reasons"]:
                    values = clean_run(records, state, m, decisions, source)
                    if not state["reasons"]:
                        kept.write(jd([rid, values]) + "\n")
                        kept_ids.append((rid, state["asd"]))
                        feature_ids.update(tid for tid, value in values.items() if abundance(value) > 0)
                if group_index % 2000 == 0:
                    LOG.info("Available profiles checked: %s", f"{group_index:,}")
        check(observed == collections.Counter(catalogue_counts), "Abundance records do not reconcile with catalogue")
        check(sum(observed.values()) == summary["counts"]["species_records"], "Abundance total differs from extraction summary")
        for rid, m in manifest.items():
            check((int(m["records_exported"]) > 0) == (rid in processed), "Profile record coverage differs: " + rid)
            if m["status"] in AVAILABLE and rid not in processed:
                states[rid]["reasons"].append("no_abundance_records")
        headers = feature_headers(feature_ids, registry)
        features = list(headers)
        matrix_header = ["run_id"] + list(headers.values())
        check(len(matrix_header) == len(set(matrix_header)), "Matrix column collision")
        LOG.info("Stage 4/5: write the main matrix and subsets (%s runs x %s features)", f"{len(kept_ids):,}", len(features))
        with contextlib.ExitStack() as stack:
            writers = [csv.writer(stack.enter_context(gzip_text(work / name, args.compression_level)), delimiter="\t", lineterminator="\n")
                       for name in (DATA_FILES[1], DATA_FILES[3], DATA_FILES[4])]
            for writer in writers:
                writer.writerow(matrix_header)
            with gzip.open(intermediate, "rt", encoding="utf-8") as f:
                for index, line in enumerate(f, 1):
                    rid, values = json.loads(line)
                    record = [rid] + [values.get(str(tid), "0") for tid in features]
                    writers[0].writerow(record)
                    if states[rid]["asd"] == "ASD_label_observed":
                        writers[1].writerow(record)
                    elif states[rid]["asd"] == "no_ASD_label_observed":
                        writers[2].writerow(record)
                    if index % 5000 == 0:
                        LOG.info("Matrix rows written: %s", f"{index:,}")
        reason_counts, metadata_count = collections.Counter(), 0
        with gzip_text(work / DATA_FILES[0], args.compression_level) as f:
            writer = None
            for row in read_tsv(source / SOURCE_FILES[0], {"run_id"}):
                rid, original_columns = row["run_id"], list(row)
                if writer is None:
                    check(not set(original_columns) & set(AUDIT_COLUMNS), "Original metadata uses reserved prep_ column names")
                    writer = csv.DictWriter(f, fieldnames=original_columns + list(AUDIT_COLUMNS), delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                row.update(metadata_additions(states[rid], manifest[rid]))
                writer.writerow(row)
                metadata_count += 1
                reason_counts.update(set(states[rid]["reasons"]))
        check(metadata_count == len(states), "Master metadata row count differs")
        LOG.info("Stage 5/5: verify outputs, input hashes and row accounting")
        counts = validate_matrices(work, matrix_header, kept_ids, feature_ids)
        check(counts["qc_pass"] == len(kept_ids), "Main matrix count differs")
        expected_failed_records = sum(int(m["records_exported"]) for rid, m in manifest.items() if states[rid]["qc"] == "fail")
        check(failed_records == expected_failed_records, "QC-failed record accounting differs")
        check(fingerprints(source, SOURCE_FILES, args.workers) == source_hashes, "Original extraction changed during preparation")
        check((source / "extraction_summary.json").read_bytes() == original_summary_bytes, "Extraction summary changed during preparation")
        outputs = fingerprints(work, DATA_FILES, args.workers)
        mapping = [{"column_index": index, "column_name": headers[tid], **registry[tid]} for index, tid in enumerate(features, 1)]
        report = {"script_version": VERSION, "script_sha256": fingerprint(Path(__file__).resolve())["sha256"],
                  "finished_at": now(), "status": "complete" if kept_ids and features else "no_eligible_matrix_rows",
                  "source": str(source), "source_files": source_hashes, "parameters": parameters,
                  "runs_in_extraction": len(states), "input_abundance_records": sum(observed.values()),
                  "qc_counts": dict(qc_counts), "ASD_annotation_counts_all_runs": dict(asd_counts),
                  "exclusion_reason_counts": dict(reason_counts), "reason_counts_overlap": True,
                  "matrix": {"runs": len(kept_ids), "species_columns": len(features), "id_column": "run_id",
                             "feature_identity": "Original GMrepo TaxID; displayed as a supplied species name",
                             "taxonomy_status": "GMrepo-reported species-profile features; not externally validated",
                             "abundance_units": "as reported; no rescaling, normalization or rounding",
                             "zero_meaning": "not reported for that feature in an available, usable profile; not proof of biological absence"},
                  "subset_counts": {"ASD": counts["ASD"], "no_ASD_annotation": counts["no_ASD_annotation"], "ambiguous_ASD_main_only": counts["ambiguous_ASD"]},
                  "qc_failed_export": {"runs_with_records": len(failed_profile_ids), "records": failed_records,
                                       "QC_failed_runs_without_records": qc_counts["fail"] - len(failed_profile_ids),
                                       "format": "Original long-table rows and columns, unchanged values and duplicates"},
                  "ASD_rules": {"mesh_ids": sorted(ASD_CODES), "phenotype_text_check": True,
                                "ambiguous": "Negated/control/sibling ASD terms: main matrix only, if otherwise eligible",
                                "main_matrix_excludes_ASD": False},
                  "local_catalogue_decisions": dict(catalogue_decisions),
                  "named_catalogue_taxids": len(registry),
                  "named_taxids_omitted_with_no_positive_value_in_main_matrix": len(registry) - len(features),
                  "duplicate_records_collapsed": sum(s["collapsed"] for s in states.values()),
                  "feature_map": mapping, "master_audit_columns": list(AUDIT_COLUMNS), "outputs": outputs,
                  "important": [
                    "All original metadata columns and every original metadata run are retained in all_metadata.",
                    "The main matrix includes ASD, no-ASD-annotation and ambiguous-ASD groups among eligible QC-passed runs.",
                    "Non-ASD annotation absence is not a verified healthy control or a clinical negative.",
                    "Local feature exclusions: nonpositive/invalid IDs, missing names, generic unknown names and explicit non-species reported ranks.",
                    "No NCBI requests, taxonomy validation, genus-table collision filtering or merging of different TaxIDs was performed.",
                    "Repeated identical measurements require original raw-response proof; conflicting/unproven repeats hold the run for review.",
                    "Runs without usable profiles are not fabricated as zero-filled samples. Their metadata remains available.",
                    "All three matrices share identical columns. Subset values are identical to their main-matrix rows.",
                    "run_id is an identifier, not a numerical model feature. Labels remain in metadata.",
                    "No prevalence/rare-taxon threshold, log/CLR transform, scaling, splitting, random masking or training was performed.",
                    "Use original project/sample/subject identifiers to design evaluation groups before training."]}
        # Final summary commits last; an interrupted publication is marked incomplete.
        atomic_json(out / RESULT_FILES[-1], {"script_version": VERSION, "status": "publishing", "started_at": now()})
        for name in DATA_FILES:
            os.replace(work / name, out / name)
        atomic_json(out / RESULT_FILES[-1], report)
    LOG.info("Finished: %s runs x %s features; ASD=%s, no-ASD-annotation=%s, ambiguous main-only=%s",
             len(kept_ids), len(features), counts["ASD"], counts["no_ASD_annotation"], counts["ambiguous_ASD"])
    LOG.info("Five data tables + filtering_summary.json: %s", out)
    LOG.info("Exclusion reasons (overlap): %s", dict(reason_counts))
    return 0 if kept_ids and features else 2


def prepare(args):
    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else source / "prepared_metagenomic"
    check(source.is_dir() and out != source, "Use an existing extraction folder and a separate output folder")
    summary_path = source / "extraction_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    check(summary.get("extraction_requests_complete") is True and summary.get("inventory_checks_passed") is True,
          "The extraction must pass its request/inventory audit")
    LOG.info("Stage 1/5: verify the four original file hashes (%s maximum I/O workers)", args.workers)
    source_hashes = fingerprints(source, SOURCE_FILES, args.workers)
    check(source_hashes == summary.get("files"), "Extraction file checksum/size mismatch")
    check(summary_bytes == summary_path.read_bytes(), "Extraction summary changed during input verification")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    # Directory locking creates no extra lock file on Linux.
    lock_fd = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another preparation process is using this output folder") from None
        parameters = {"compression_level": args.compression_level, "external_taxonomy_mapping": False,
                      "all_eligible_QC_passed_phenotypes": True, "headers": "species_names_with_ID_suffix_only_for_collisions"}
        previous = out / RESULT_FILES[-1]
        if previous.is_file():
            try:
                report = json.loads(previous.read_text())
                same = (report.get("status") == "complete" and report.get("script_version") == VERSION
                        and report.get("parameters") == parameters and report.get("source_files") == source_hashes
                        and report.get("script_sha256") == fingerprint(Path(__file__).resolve())["sha256"])
                if same and fingerprints(out, DATA_FILES, args.workers) == report.get("outputs"):
                    LOG.info("Existing outputs are complete and match all hashes; no rewrite needed")
                    LOG.info("Main matrix: %s runs x %s features; output: %s", report["matrix"]["runs"], report["matrix"]["species_columns"], out)
                    return 0
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return build(source, out, source_hashes, summary, args, parameters, summary_bytes)
    finally:
        os.close(lock_fd)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", type=Path, default=Path("metagenomic_totla"), help="Completed original extraction folder")
    p.add_argument("--out", type=Path, help="Result folder; default SOURCE/prepared_metagenomic")
    p.add_argument("--workers", type=int, default=10, help="Maximum workers for independent checksum I/O; table processing streams locally")
    p.add_argument("--compression-level", type=int, default=4, help="Gzip level: 1 faster/larger, 9 slower/smaller")
    p.add_argument("--version", action="version", version=VERSION)
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 32 or not 1 <= args.compression_level <= 9:
        p.error("Use 1-32 workers and compression level 1-9")
    return args


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv.field_size_limit(32 * 1024 * 1024)
    LOG.info("GMrepo preparation %s: all eligible QC-passed phenotypes, local data only", VERSION)
    try:
        return prepare(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted. Use results only when filtering_summary.json status is complete and hashes match.")
        return 130
    except Exception as exc:
        LOG.error("Preparation stopped: %s: %s", type(exc).__name__, exc)
        LOG.error("Use results only when filtering_summary.json status is complete and hashes match.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
