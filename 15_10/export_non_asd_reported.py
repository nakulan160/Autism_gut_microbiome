#!/usr/bin/env python3
"""Export original GMrepo abundances with run-ID rows and reported-name columns.

Python 3.10+; standard library only. No packages or network access are needed.

Selection: QC pass (1), no ASD-related annotation, available metagenomic profile.
Unreported entries are EMPTY TSV fields. Reported values are copied as text,
so reported zeros, decimal precision, names, and Unknown entries are preserved.
No abundance normalization, scaling, rounding, taxonomy filtering, or imputation.

Feature identity is (original TaxID, original scientific name, reported rank).
Different reported names remain separate. Names shared by different identities
get identifying suffixes in the header. Identical repeated records occupy one
matrix cell with their unchanged value. Conflicting repeated values stop export.
The input files are never overwritten. Existing output files are refused.
"""

from __future__ import annotations

import argparse
import collections
import csv
from decimal import Decimal, InvalidOperation
import gzip
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata


ASD_CODES = {"D000067877", "D001321", "D020817"}
ASD_WORDS = re.compile(r"\b(?:autism|autistic|asd|asperger(?:s)?)\b", re.I)
MESH_CODES = re.compile(r"\b[DC][0-9]{6,9}\b")
METAGENOMIC_TYPES = {
    "metagenomics", "metagenomic", "shotgun metagenomics",
    "metagenomic shotgun", "whole metagenome shotgun", "wgs",
}
AVAILABLE_PROFILES = {"downloaded", "downloaded_with_flags"}
FEATURE_FIELDS = ("ncbi_taxon_id", "scientific_name", "reported_rank")


def open_text(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def scalars(value):
    """Read the extractor's scalar or nested-list metadata representations."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from scalars(item)
    elif value is not None and str(value).strip():
        text = str(value).strip()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except ValueError:
                pass
            else:
                if isinstance(decoded, list):
                    yield from scalars(decoded)
                    return
        yield text


def normalized(value):
    return re.sub(
        r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", str(value)).casefold()
    ).strip()


def require_fields(actual, required, path):
    missing = sorted(set(required) - set(actual or []))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")


def qc_pass(values):
    canonical = set()
    for value in values:
        try:
            number = Decimal(value)
            text = str(int(number)) if number.is_finite() and number == int(number) else value
        except (InvalidOperation, ValueError, OverflowError):
            text = value
        canonical.add(text)
    # The existing GMrepo preparation convention: 1=pass, 0=fail.
    # A missing, unrecognized, or conflicting QC annotation is not a pass.
    return canonical == {"1"}


def select_runs(metadata):
    required = {
        "run_id", "qc_status_values", "phenotype_mesh_ids", "phenotype_names",
        "asd_label_observed", "profile_status", "all_observed_metadata_json",
    }
    all_ids, selected = set(), set()
    excluded = collections.Counter()
    with open_text(metadata) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, required, metadata)
        for row in reader:
            rid = row["run_id"]
            if not rid or rid in all_ids:
                raise ValueError(f"Empty or repeated metadata run_id: {rid!r}")
            all_ids.add(rid)
            observed = json.loads(row["all_observed_metadata_json"])
            if not isinstance(observed, dict):
                raise ValueError(f"Invalid observed-metadata object for {rid}")
            qc = set(scalars(json.loads(row["qc_status_values"])))
            mesh = set(scalars(json.loads(row["phenotype_mesh_ids"])))
            terms = set(scalars(json.loads(row["phenotype_names"])))
            sequencing = set()
            for key, values in observed.items():
                key = key.casefold()
                values = set(scalars(values))
                if key == "qcstatus":
                    qc.update(values)
                if key in {"experiment_type", "library_strategy"}:
                    sequencing.update(values)
                if key in {"disease", "mesh_id", "diseases", "_associated_phenotype_mesh_id"}:
                    for value in values:
                        mesh.update(MESH_CODES.findall(value))
                if key in {"phenotype", "phenotypes", "term", "_associated_phenotype_name"}:
                    terms.update(values)
            qc.update(scalars(row.get("QCStatus")))
            sequencing.update(scalars(row.get("experiment_type")))
            if not qc_pass(qc):
                excluded["QC_failed_missing_or_conflicting"] += 1
            elif row["profile_status"] not in AVAILABLE_PROFILES:
                excluded["no_available_profile"] += 1
            elif any(normalized(value) not in METAGENOMIC_TYPES for value in sequencing):
                excluded["conflicting_sequencing_annotation"] += 1
            elif (
                row["asd_label_observed"].casefold() == "yes"
                or mesh & ASD_CODES
                or any(ASD_WORDS.search(normalized(term)) for term in terms)
            ):
                # This also withholds ambiguous ASD/control/sibling descriptions,
                # matching the earlier no-ASD-annotation subset definition.
                excluded["ASD_related_annotation"] += 1
            else:
                selected.add(rid)
    if not selected:
        raise ValueError("No non-ASD-annotated, QC-passed runs with profiles were found.")
    print(f"Selected runs: {len(selected):,}", flush=True)
    print("Metadata exclusions:", dict(excluded), flush=True)
    return all_ids, selected


def profiles(abundance, all_ids, selected):
    """Stream one run at a time; the original extractor groups rows by run ID."""
    required = {"run_id", "relative_abundance", *FEATURE_FIELDS}
    seen_runs = set()
    with open_text(abundance) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        require_fields(header, required, abundance)
        if len(header) != len(set(header)):
            raise ValueError("The source long table contains repeated column headers.")
        index = {name: header.index(name) for name in required}
        for rid, records in itertools.groupby(reader, key=lambda row: row[index["run_id"]]):
            if rid in seen_runs:
                raise ValueError(f"Source rows are not grouped by run_id: {rid}")
            seen_runs.add(rid)
            if rid not in all_ids:
                raise ValueError(f"Abundance run_id has no metadata record: {rid}")
            if rid not in selected:
                continue
            values = {}
            repeated = 0
            for row in records:
                if len(row) != len(header):
                    raise ValueError(f"Wrong source row width for run {rid}")
                key = tuple(row[index[field]] for field in FEATURE_FIELDS)
                value = row[index["relative_abundance"]]
                if key in values:
                    if values[key] != value:
                        raise ValueError(
                            f"Conflicting repeated entry: run={rid}, feature={key!r}, "
                            f"values={values[key]!r} and {value!r}. No summing or overwriting."
                        )
                    repeated += 1
                else:
                    # Keep the ORIGINAL text, including any explicitly reported 0.
                    values[key] = value
            yield rid, values, repeated


def make_headers(features):
    names = collections.Counter(key[1] for key in features)
    used = {"run_id"}
    headers = []
    for tid, name, rank in features:
        label = name
        if not label or names[name] > 1 or label == "run_id":
            label = f"{name or '(unnamed)'} [taxid={tid}; rank={rank}]"
        base, suffix = label, 2
        while label in used:
            label = f"{base} [column={suffix}]"
            suffix += 1
        used.add(label)
        headers.append(label)
    return headers


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def export(metadata, abundance, output):
    metadata, abundance, output = (Path(p).expanduser().resolve() for p in (metadata, abundance, output))
    if output in {metadata, abundance}:
        raise ValueError("Output must be separate from the original source files.")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a different --output: {output}")
    source_signatures = {p: signature(p) for p in (metadata, abundance)}
    all_ids, selected = select_runs(metadata)
    features, found = set(), set()
    reported_records = repeated_records = unique_cells = 0
    print("Pass 1/2: inspect original reported entries and check repeated records", flush=True)
    for rid, values, repeated in profiles(abundance, all_ids, selected):
        found.add(rid)
        features.update(values)
        repeated_records += repeated
        unique_cells += len(values)
        reported_records += len(values) + repeated
    missing = selected - found
    if missing:
        raise ValueError(f"{len(missing)} selected runs have no source records; examples: {sorted(missing)[:5]}")
    features = sorted(features, key=lambda key: (key[1].casefold(), key[0], key[2]))
    headers = make_headers(features)
    if not features:
        raise ValueError("No reported features were found.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".partial-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    written = 0
    print(f"Pass 2/2: write {len(selected):,} rows x {len(features):,} feature columns", flush=True)
    try:
        opener = gzip.open if str(output).endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["run_id", *headers])
            for rid, values, _ in profiles(abundance, all_ids, selected):
                # The missing-entry default is an EMPTY STRING, never zero.
                writer.writerow([rid, *(values.get(key, "") for key in features)])
                written += 1
                if written % 5000 == 0:
                    print(f"Written {written:,}/{len(selected):,} runs", flush=True)
        if written != len(selected):
            raise ValueError("Export row count differs from the selected run count.")
        if any(signature(path) != before for path, before in source_signatures.items()):
            raise ValueError("A source file changed during export; output was not installed.")
        # The temporary file is on the same filesystem. Linking atomically installs
        # it while refusing to overwrite even an output created during this run.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved: {output}", flush=True)
    print(f"Rows: {written:,}; feature columns: {len(features):,}; total columns: {len(features)+1:,}")
    print(f"Original selected records: {reported_records:,}; populated cells: {unique_cells:,}")
    print(f"Identical repeated records represented in existing cells: {repeated_records:,}")
    print("Unreported cells: blank. Reported abundance text: unchanged.")
    print("Different reported names: separate columns. No taxonomy filtering; Unknown is retained if reported.")
    print("Non-ASD means no ASD-related annotation observed, not verified healthy controls.")
    return {"runs": written, "features": len(features), "reported_records": reported_records,
            "populated_cells": unique_cells, "identical_repeats": repeated_records}


def main():
    csv.field_size_limit(sys.maxsize)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("metagenomic_totla"), help="Original extraction directory")
    parser.add_argument("--metadata", type=Path, help="Override original metadata path")
    parser.add_argument("--abundance", type=Path, help="Override original long-abundance path")
    parser.add_argument("--output", type=Path, help="New .tsv or .tsv.gz file; existing files are refused")
    args = parser.parse_args()
    metadata = args.metadata or args.source / "all_metagenomic_metadata.tsv"
    abundance = args.abundance or args.source / "all_species_abundance_long.tsv.gz"
    output = args.output or args.source / "non_asd_qc_pass_reported_blank.tsv.gz"
    try:
        export(metadata, abundance, output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
