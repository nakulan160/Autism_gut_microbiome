#!/usr/bin/env python3
"""Download GMrepo metagenomic species records without biological filtering.

Python 3.10+; standard library only. Run again with the same output directory
to resume. A GPU is not used. See --help for diagnostic and recovery options.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import csv
import dataclasses
import datetime as dt
import decimal
import email.utils
import gzip
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

VERSION = "1.0.0"
ASD = "D000067877"
FILTER_SQL = "experiment_type = 'Metagenomics'"
RUN_RE = re.compile(r"^(?:SRR|ERR|DRR)[0-9]+$")
MESH_RE = re.compile(r"\b[DC][0-9]{6,9}\b")
STOP = threading.Event()
LOG = logging.getLogger("gmrepo")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":"), allow_nan=False)


def loads(x):
    # Decimal preserves the digits reported in JSON abundance values.
    return json.loads(x, parse_float=decimal.Decimal,
                      parse_constant=lambda s: s)


def digest(x):
    return hashlib.sha256(x).hexdigest()


def atomic_bytes(path, content, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temp.open("wb") as f:
            if compressed:
                with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
                    g.write(content)
            else:
                f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def atomic_json(path, obj):
    atomic_bytes(path, (dumps(obj) + "\n").encode("utf-8"))


def run_id(row):
    if not isinstance(row, dict):
        value = str(row).strip()
        return value if RUN_RE.fullmatch(value) else None
    for k in ("run_id", "run_accession", "RunID", "Run ID", "accession_id"):
        value = str(row.get(k, "")).strip()
        if RUN_RE.fullmatch(value):
            return value
    return None


def metagenomic(row):
    text = str(row.get("experiment_type", row.get("library_strategy", "")))
    text = re.sub(r"[\s_-]+", " ", text.strip().casefold())
    return text in {"metagenomics", "metagenomic", "shotgun metagenomics",
                    "metagenomic shotgun", "whole metagenome shotgun", "wgs"}


def records(obj, keys=("runs", "data", "results")):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in keys:
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("Unexpected response shape; expected an array of records")


def nonnegative_int(obj, keys):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            if isinstance(value, bool):
                continue
            try:
                number = decimal.Decimal(str(value))
                if number.is_finite() and number >= 0 and number == int(number):
                    return int(number)
            except (ValueError, TypeError, decimal.InvalidOperation):
                pass
    raise ValueError("Missing or invalid count: " + ", ".join(keys))


def cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return dumps(value)
    return str(value)


@dataclasses.dataclass
class Reply:
    key: str
    endpoint: str
    payload: dict
    path: str = ""
    sha256: str = ""
    fetched_at: str = ""
    data: object = None
    error: str = ""
    attempts: int = 0
    cached: bool = False


class API:
    def __init__(self, root, args):
        self.root, self.args = Path(root), args
        self.base = args.base_url.rstrip("/") + "/"
        self.rate_lock = threading.Lock()
        self.next_request = 0.0

    def key(self, endpoint, payload):
        return digest(dumps([self.base, endpoint, payload]).encode())

    def throttle(self):
        with self.rate_lock:
            now = time.monotonic()
            wait = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + 1 / self.args.requests_per_second
        if STOP.wait(wait):
            raise InterruptedError("Stopped")

    def request(self, endpoint, payload=None, *, profile=None, force=False):
        payload = {} if payload is None else payload
        key = self.key(endpoint, payload)
        part = "profiles" if profile else "inventory"
        name = profile if profile else endpoint + "-" + key
        path = self.root / "raw_api_responses" / part / (name + ".json.gz")
        reply = Reply(key, endpoint, payload, str(path.relative_to(self.root)))
        if path.exists() and not force:
            try:
                with gzip.open(path, "rb") as f:
                    raw = f.read()
                reply.data = loads(raw)
                if not isinstance(reply.data, (dict, list)):
                    raise ValueError("JSON is not an object or array")
                reply.sha256 = digest(raw)
                reply.fetched_at = dt.datetime.fromtimestamp(
                    path.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
                reply.cached = True
                return reply
            except (OSError, EOFError, ValueError, UnicodeError):
                bad = path.with_name(path.name + ".corrupt-" + uuid.uuid4().hex[:8])
                os.replace(path, bad)
                LOG.warning("Corrupt cached response moved to %s", bad.name)
        for attempt in range(1, self.args.retries + 1):
            if STOP.is_set():
                break
            reply.attempts = attempt
            delay = min(45.0, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
            transient = True
            raw = b""
            try:
                self.throttle()
                request = urllib.request.Request(
                    self.base + endpoint.strip("/") + "/",
                    data=dumps(payload).encode("utf-8"), method="POST",
                    headers={"Content-Type": "application/json", "Accept": "application/json",
                             "Accept-Encoding": "gzip", "Cache-Control": "no-cache",
                             "User-Agent": "GMrepoSpeciesExtraction/" + VERSION})
                with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                    raw = response.read(self.args.max_response_mb * 1024 * 1024 + 1)
                    if len(raw) > self.args.max_response_mb * 1024 * 1024:
                        raise ValueError("Response exceeds --max-response-mb")
                    if response.headers.get("Content-Encoding", "").lower() == "gzip":
                        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                            raw = f.read(self.args.max_response_mb * 1024 * 1024 + 1)
                        if len(raw) > self.args.max_response_mb * 1024 * 1024:
                            raise ValueError("Decompressed response exceeds size limit")
                obj = loads(raw)
                if not isinstance(obj, (dict, list)):
                    raise ValueError("Expected a JSON object or array; received other content")
                if isinstance(obj, dict):
                    code = str(obj.get("code", "200"))
                    if (code.isdigit() and int(code) >= 400) or obj.get("error"):
                        raise ValueError("API error response: " + str(obj)[:250])
                if path.exists():
                    with gzip.open(path, "rb") as f:
                        old = f.read()
                    if digest(old) != digest(raw):
                        history = self.root / "raw_api_responses" / "history"
                        dest = history / (name + "-" + digest(old) + ".json.gz")
                        if not dest.exists():
                            atomic_bytes(dest, old, compressed=True)
                atomic_bytes(path, raw, compressed=True)
                reply.data, reply.sha256, reply.fetched_at = obj, digest(raw), utc()
                reply.error = ""
                return reply
            except InterruptedError:
                break
            except urllib.error.HTTPError as exc:
                raw = exc.read(262144)
                reply.error = "HTTP " + str(exc.code) + ": " + str(exc.reason)
                transient = exc.code in {408, 425, 429, 500, 502, 503, 504}
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        with contextlib.suppress(Exception):
                            stamp = email.utils.parsedate_to_datetime(retry_after)
                            delay = max(delay, stamp.timestamp() - time.time())
            except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                    EOFError, UnicodeError) as exc:
                reply.error = type(exc).__name__ + ": " + str(exc)[:350]
            if raw:
                error_path = self.root / "raw_api_responses" / "errors" / (
                    key + "-" + digest(raw) + ".body.gz")
                if not error_path.exists():
                    atomic_bytes(error_path, raw, compressed=True)
            if not transient or attempt == self.args.retries:
                break
            STOP.wait(min(300.0, max(0.0, delay)))
        reply.error = reply.error or "Interrupted before request completed"
        return reply


class State:
    def __init__(self, root, base):
        self.root = Path(root)
        self.db = sqlite3.connect(self.root / "extraction_checkpoint.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS config(k TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS memberships(
          run_id TEXT, source TEXT, PRIMARY KEY(run_id,source));
        CREATE TABLE IF NOT EXISTS observations(
          run_id TEXT, sha TEXT, body TEXT, PRIMARY KEY(run_id,sha));
        CREATE TABLE IF NOT EXISTS sources(
          run_id TEXT, observation_sha TEXT, request_key TEXT,
          PRIMARY KEY(run_id,observation_sha,request_key));
        CREATE TABLE IF NOT EXISTS calls(
          request_key TEXT PRIMARY KEY, endpoint TEXT, payload TEXT, status TEXT,
          raw_file TEXT, sha256 TEXT, fetched_at TEXT, error TEXT, row_count INTEGER);
        CREATE TABLE IF NOT EXISTS seen(
          source_group TEXT, run_id TEXT, PRIMARY KEY(source_group,run_id));
        CREATE TABLE IF NOT EXISTS profiles(
          run_id TEXT PRIMARY KEY, status TEXT, raw_file TEXT, sha256 TEXT,
          fetched_at TEXT, error TEXT, records INTEGER, flags TEXT);
        """)
        existing = self.get("base_url")
        if existing and existing != base:
            raise ValueError("This directory belongs to a different API base URL; use a new directory")
        self.set("base_url", base)
        if not self.get("created_at"):
            self.set("created_at", utc())
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM config WHERE k=?", (key,)).fetchone()
        return loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO config VALUES(?,?)", (key, dumps(value)))

    def call(self, reply, status="ok", row_count=None, error=""):
        self.db.execute("INSERT OR REPLACE INTO calls VALUES(?,?,?,?,?,?,?,?,?)", (
            reply.key, reply.endpoint, dumps(reply.payload),
            "failed" if reply.error else status, reply.path, reply.sha256,
            reply.fetched_at, reply.error or error, row_count))

    def add_run(self, rid, source):
        self.db.execute("INSERT OR IGNORE INTO runs VALUES(?)", (rid,))
        self.db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?)", (rid, source))

    def observe(self, rid, row, request_key):
        text = dumps(row)
        sha = digest(text.encode())
        self.db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?)", (rid, sha, text))
        self.db.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?)", (rid, sha, request_key))

    def ingest(self, reply, group, selected=False, phenotype=None):
        rows = records(reply.data)
        bad = 0
        for row in rows:
            rid = run_id(row)
            if not rid or not isinstance(row, dict):
                bad += 1
                continue
            self.db.execute("INSERT OR IGNORE INTO seen VALUES(?,?)", (group, rid))
            known = self.db.execute("SELECT 1 FROM runs WHERE run_id=?", (rid,)).fetchone()
            if phenotype:
                row = dict(row, _associated_phenotype_mesh_id=phenotype[0],
                           _associated_phenotype_name=phenotype[1])
            # Preserve observations even if another inventory discovers this ID later.
            # Non-metagenomic IDs are not added to runs or exported as profiles.
            self.observe(rid, row, reply.key)
            if selected and not known and not metagenomic(row):
                bad += 1
                continue
            if selected or metagenomic(row) or known:
                self.add_run(rid, group)
        self.call(reply, "schema_error" if bad else "ok", len(rows),
                  str(bad) + " records with invalid IDs or unexpected sequencing type" if bad else "")
        return len(rows), bad


def parallel(jobs, fn, workers):
    """Bounded futures: large downloads do not accumulate in memory."""
    iterator = iter(jobs)
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    pending = {}
    try:
        for _ in range(workers):
            job = next(iterator, None)
            if job is not None:
                pending[pool.submit(fn, job)] = job
        while pending and not STOP.is_set():
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                yield job, future.result()
                nxt = next(iterator, None)
                if nxt is not None and not STOP.is_set():
                    pending[pool.submit(fn, nxt)] = nxt
    except KeyboardInterrupt:
        STOP.set()
        raise
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def inventory(api, state, args):
    if state.get("inventory_finished", False):
        LOG.info("Using the frozen inventory from %s", state.get("created_at"))
        return
    LOG.info("Stage 1/3: collect independent inventories and full phenotype metadata")
    problems = []

    def obtain(endpoint, payload=None):
        key = api.key(endpoint, payload or {})
        prev = state.db.execute("SELECT status FROM calls WHERE request_key=?", (key,)).fetchone()
        r = api.request(endpoint, payload, force=bool(prev and prev[0] != "ok"))
        if r.error:
            state.call(r)
            state.db.commit()
            raise RuntimeError(endpoint + ": " + r.error)
        state.call(r, "pending_validation")
        state.db.commit()
        return r

    stats_reply = obtain("getAllProjectsWithMetadataAndSomeStatistics")
    stats = stats_reply.data.get("stats", {}) if isinstance(stats_reply.data, dict) else {}
    total_global = nonnegative_int(stats, ("all_runs_count",))
    state.call(stats_reply)
    state.set("database_statistics", stats_reply.data)

    count_reply = obtain("runSelectorCount", {"sql": FILTER_SQL})
    expected_selector = nonnegative_int(count_reply.data, ("total",))
    state.call(count_reply)
    ids_reply = obtain("runSelectorUniqRunID", {"sql": FILTER_SQL})
    id_rows = records(ids_reply.data)
    selector_ids = {run_id(x) for x in id_rows}
    invalid_ids = None in selector_ids
    selector_ids.discard(None)
    if invalid_ids or len(selector_ids) != expected_selector:
        problems.append("Selector count differs from valid unique ID list")
    for rid in selector_ids:
        state.add_run(rid, "selector_ids")
    state.call(ids_reply, "schema_error" if invalid_ids or len(selector_ids) != expected_selector else "ok", len(id_rows))
    LOG.info("Selector inventory: %s distinct metagenomic runs", f"{len(selector_ids):,}")

    all_reply = obtain("runSelectorAll", {"sql": FILTER_SQL})
    state.ingest(all_reply, "selector_metadata", selected=True)
    registry_reply = obtain("get_all_phenotypes")
    phenotypes = records(registry_reply.data, ("phenotypes",))
    registry = {}
    for p in phenotypes:
        if not isinstance(p, dict):
            raise ValueError("Invalid phenotype registry row")
        mesh = p.get("disease", p.get("mesh_id", p.get("uid")))
        if mesh:
            registry[str(mesh)] = str(p.get("term", p.get("phenotype", "")))
    for mesh, term in stats_reply.data.get("disease2term", {}).items():
        if mesh:
            registry.setdefault(str(mesh), str(term))
    if not registry:
        raise ValueError("Phenotype registry is empty")
    state.call(registry_reply, row_count=len(phenotypes))
    state.set("phenotype_registry", registry)
    state.db.commit()

    counts = {}
    jobs = list(registry)
    invalid_cached_calls = {r[0] for r in state.db.execute(
        "SELECT request_key FROM calls WHERE status!='ok'")}
    def count_job(mesh):
        payload = {"mesh_id": mesh}
        endpoint = "countAssociatedRunsByPhenotypeMeshID"
        return api.request(endpoint, payload, force=api.key(endpoint, payload) in invalid_cached_calls)
    for mesh, reply in parallel(jobs, count_job, args.workers):
        try:
            if reply.error:
                raise ValueError(reply.error)
            counts[mesh] = nonnegative_int(reply.data, ("nr_assoc_runs", "total"))
            state.call(reply)
        except ValueError as exc:
            problems.append("Phenotype count unavailable: " + mesh)
            state.call(reply, "schema_error", error=str(exc))
            # Still request page 0 and continue until an empty/short page.
            counts[mesh] = 0
        state.db.commit()
    state.set("phenotype_counts", counts)

    groups = {"global": {"endpoint": "getAllRunsAsync", "expected": total_global,
                          "base": {}, "phenotype": None}}
    for mesh in registry:
        groups["phenotype:" + mesh] = {
            "endpoint": "getAssociatedRunsByPhenotypeMeshIDLimit", "expected": counts[mesh],
            "base": {"mesh_id": mesh}, "phenotype": (mesh, registry[mesh])}
    stats_by_group = {k: {"rows": 0, "last_page_rows": None, "failed_pages": 0}
                      for k in groups}
    jobs = []
    for group, spec in groups.items():
        pages = math.ceil(spec["expected"] / args.page_size) + 1
        if pages > args.max_pages:
            problems.append("Advertised source count exceeds --max-pages: " + group)
        for page in range(max(1, min(pages, args.max_pages))):
            jobs.append((group, page))

    def page_job(job):
        group, page = job
        spec = groups[group]
        payload = dict(spec["base"], limit=args.page_size, skip=page * args.page_size)
        return api.request(spec["endpoint"], payload,
                           force=api.key(spec["endpoint"], payload) in invalid_cached_calls)

    processed = 0
    extensions = []
    for job, reply in parallel(jobs, page_job, args.workers):
        group, page = job
        spec, info = groups[group], stats_by_group[group]
        try:
            if reply.error:
                raise ValueError(reply.error)
            n, bad = state.ingest(reply, group, phenotype=spec["phenotype"])
            info["rows"] += n
            if bad:
                problems.append("Unidentified run records in " + group)
            if page == math.ceil(spec["expected"] / args.page_size):
                info["last_page_rows"] = n
                if n >= args.page_size:
                    extensions.append((group, page + 1))
        except ValueError as exc:
            state.call(reply, "schema_error", error=str(exc))
            info["failed_pages"] += 1
            problems.append("Inventory page unresolved: " + group + "/" + str(page))
        processed += 1
        state.db.commit()
        if processed % 25 == 0 or processed == len(jobs):
            LOG.info("Metadata pages processed: %s/%s", processed, len(jobs))

    # Extend beyond stale counts until each source returns a short page.
    for group, page in extensions:
        stalled_pages = 0
        while page < args.max_pages:
            reply = page_job((group, page))
            info, spec = stats_by_group[group], groups[group]
            try:
                if reply.error:
                    raise ValueError(reply.error)
                before = state.db.execute("SELECT COUNT(*) FROM seen WHERE source_group=?", (group,)).fetchone()[0]
                n, bad = state.ingest(reply, group, phenotype=spec["phenotype"])
                after = state.db.execute("SELECT COUNT(*) FROM seen WHERE source_group=?", (group,)).fetchone()[0]
                info["rows"] += n
                info["last_page_rows"] = n
                state.db.commit()
                if bad:
                    problems.append("Unidentified records in extended page: " + group)
                if n < args.page_size:
                    break
                stalled_pages = stalled_pages + 1 if after == before else 0
                if stalled_pages >= 2:
                    problems.append("Repeated full pages beyond advertised count: " + group)
                    break
            except ValueError as exc:
                state.call(reply, "schema_error", error=str(exc))
                info["failed_pages"] += 1
                problems.append("Extended inventory page failed: " + group)
                break
            page += 1
        else:
            problems.append("Maximum page safeguard reached: " + group)

    for group, spec in groups.items():
        info = stats_by_group[group]
        info["unique_runs_all_sequencing_types"] = state.db.execute(
            "SELECT COUNT(*) FROM seen WHERE source_group=?", (group,)).fetchone()[0]
        info["expected_source_count"] = spec["expected"]
        if info["rows"] < spec["expected"]:
            problems.append("Fewer rows than advertised: " + group)
        if group.startswith("phenotype:") and info["unique_runs_all_sequencing_types"] < spec["expected"]:
            problems.append("Phenotype unique-ID coverage below advertised count: " + group)
    missing = state.db.execute("""SELECT m.run_id FROM memberships m WHERE m.source='selector_ids'
      AND NOT EXISTS(SELECT 1 FROM memberships n WHERE n.run_id=m.run_id
                     AND n.source='selector_metadata') ORDER BY m.run_id""").fetchall()
    if missing:
        problems.append("Selector metadata missing for some expected IDs")
    state.set("inventory_audit", {"selector_advertised_count": expected_selector,
        "selector_unique_ids": len(selector_ids), "source_groups": stats_by_group,
        "selector_ids_missing_selector_metadata": [r[0] for r in missing],
        "problems": sorted(set(problems)), "finished_at": utc()})
    state.set("inventory_finished", not problems)
    state.db.commit()
    total = state.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    LOG.info("Frozen union inventory: %s metagenomic run IDs", f"{total:,}")
    if problems:
        LOG.warning("Inventory coverage has %s unresolved checks; see extraction_summary.json",
                    len(set(problems)))


def profile_info(obj, requested):
    if not isinstance(obj, dict):
        raise ValueError("Profile response is not an object")
    run_rows = obj.get("run", [])
    if isinstance(run_rows, dict):
        run_rows = [run_rows]
    if not isinstance(run_rows, list):
        raise ValueError("Unexpected run metadata shape")
    matches = [r for r in run_rows if run_id(r) == requested]
    other_ids = {run_id(r) for r in run_rows} - {requested, None}
    if other_ids:
        raise ValueError("Response includes unrelated run IDs")
    species = obj.get("species")
    if species is not None and not isinstance(species, list):
        raise ValueError("Species field is not an array; raw response retained")
    if not matches:
        if not run_rows and not species:
            return "not_found", [], [], ["no_run_metadata_returned"]
        raise ValueError("Response run ID does not match requested run ID")
    flags = []
    if any(not metagenomic(r) for r in matches if r.get("experiment_type") is not None):
        flags.append("sequencing_type_conflict_with_inventory")
    if species is None:
        return "no_species_field", matches, [], flags
    if not species:
        return "empty_species_profile", matches, [], flags
    return "downloaded", matches, species, flags


def download_profiles(api, state, args):
    LOG.info("Stage 2/3: retrieve all inventoried runs; ASD and QC labels are not filters")
    jobs = []
    for row in state.db.execute("SELECT r.run_id,p.status FROM runs r LEFT JOIN profiles p USING(run_id) ORDER BY r.run_id"):
        force = row["status"] in {"failed", "schema_error", "cache_error"}
        force = force or bool(args.retry_empty and row["status"] in
                             {"not_found", "no_species_field", "empty_species_profile"})
        jobs.append((row["run_id"], force))
    if args.max_runs:
        jobs = jobs[:args.max_runs]
    total, done, consecutive_errors = len(jobs), 0, 0
    def work(job):
        rid, force = job
        return api.request("getFullTaxonomicProfileByRunID", {"run_id": rid},
                           profile=rid, force=force)
    for job, reply in parallel(jobs, work, args.workers):
        rid = job[0]
        flags, n, error, status = [], 0, reply.error, "failed"
        if not reply.error:
            if isinstance(reply.data, dict) and isinstance(reply.data.get("species"), list):
                n = len(reply.data["species"])
            try:
                status, run_rows, species, flags = profile_info(reply.data, rid)
                n = len(species)
                for row in run_rows:
                    state.observe(rid, row, reply.key)
                phenos = reply.data.get("phenotypes", [])
                if isinstance(phenos, list):
                    for p in phenos:
                        if isinstance(p, dict):
                            state.observe(rid, {"run_id": rid,
                                "_associated_phenotype_mesh_id": p.get("disease", p.get("mesh_id")),
                                "_associated_phenotype_name": p.get("term")}, reply.key)
                state.call(reply, row_count=n)
                consecutive_errors = 0
            except ValueError as exc:
                status, error = "schema_error", str(exc)
                state.call(reply, "schema_error", error=error)
                consecutive_errors += 1
        else:
            state.call(reply)
            consecutive_errors += 1
        state.db.execute("INSERT OR REPLACE INTO profiles VALUES(?,?,?,?,?,?,?,?)", (
            rid, status, reply.path, reply.sha256, reply.fetched_at,
            error, n, dumps(flags)))
        state.db.commit()
        done += 1
        if error:
            LOG.warning("%s: %s", rid, error)
        if done % 50 == 0 or done == total:
            LOG.info("Profiles processed: %s/%s (cached files are reused)", f"{done:,}", f"{total:,}")
        if consecutive_errors >= args.max_consecutive_errors:
            LOG.error("Stopping after %s consecutive unresolved requests; rerun to resume", consecutive_errors)
            STOP.set()
            break


def merged_metadata(state, rid):
    merged = collections.defaultdict(dict)
    for row in state.db.execute("SELECT body FROM observations WHERE run_id=? ORDER BY sha", (rid,)):
        for key, value in loads(row[0]).items():
            if value is not None and value != "":
                merged[key][dumps(value)] = value
    values = {k: [v[x] for x in sorted(v)] for k, v in merged.items()}
    meshes, terms, qc = set(), set(), set()
    for key, vals in values.items():
        if key in {"disease", "mesh_id", "diseases", "_associated_phenotype_mesh_id"}:
            for val in vals:
                meshes.update(MESH_RE.findall(cell(val)))
        if key in {"phenotype", "phenotypes", "term", "_associated_phenotype_name"}:
            terms.update(cell(v) for v in vals)
        if key.casefold() == "qcstatus":
            qc.update(cell(v) for v in vals)
    return values, sorted(meshes), sorted(terms), sorted(qc)


def abundance_status(value):
    if value is None or value == "":
        return "missing", None
    try:
        n = decimal.Decimal(str(value))
        if not n.is_finite():
            return "nonfinite", None
        return ("negative" if n < 0 else "valid"), n
    except decimal.InvalidOperation:
        return "not_numeric", None


def export_files(state, args, interrupted=False):
    LOG.info("Stage 3/3: write four consolidated files from preserved responses")
    root = state.root
    names = ["all_metagenomic_metadata.tsv", "all_species_abundance_long.tsv.gz",
             "taxa_dictionary.tsv", "extraction_manifest.tsv"]
    staging = root / "export_in_progress"
    staging.mkdir(exist_ok=True)
    atomic_json(root / "extraction_summary.json", {
        "status": "export_in_progress", "updated_at": utc(), "version": VERSION})
    columns = set()
    for row in state.db.execute("SELECT body FROM observations"):
        columns.update(loads(row[0]))
    columns.discard("run_id")
    metadata_header = ["run_id", "phenotype_mesh_ids", "phenotype_names", "qc_status_values",
                       "asd_label_observed", "inventory_sources", "profile_status",
                       "fields_with_multiple_observed_values", "all_observed_metadata_json"]
    extras = [k for k in sorted(columns) if k not in metadata_header]
    species_header = ["run_id", "record_index", "ncbi_taxon_id", "scientific_name",
                      "reported_rank", "relative_abundance", "numeric_status",
                      "also_reported_in_genus_table", "duplicate_taxon_id_in_response",
                      "record_flags", "source_raw_file", "source_sha256"]
    manifest_header = ["run_id", "status", "records_returned", "records_exported",
                       "finite_numeric_abundance_sum", "invalid_numeric_records",
                       "taxa_also_reported_in_genus_table", "duplicate_taxon_records",
                       "flags", "source_raw_file", "source_sha256", "fetched_at", "error"]
    taxa = {}
    status_counts, totals = collections.Counter(), collections.Counter()
    with contextlib.ExitStack() as stack:
        meta_f = stack.enter_context((staging / names[0]).open("w", newline="", encoding="utf-8"))
        long_raw = stack.enter_context((staging / names[1]).open("wb"))
        long_gzip = stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=long_raw, mtime=0))
        long_f = stack.enter_context(io.TextIOWrapper(long_gzip, newline="", encoding="utf-8"))
        manifest_f = stack.enter_context((staging / names[3]).open("w", newline="", encoding="utf-8"))
        mw, sw, aw = [csv.writer(f, delimiter="\t", lineterminator="\n")
                      for f in (meta_f, long_f, manifest_f)]
        mw.writerow(metadata_header + extras)
        sw.writerow(species_header)
        aw.writerow(manifest_header)
        all_ids = [r[0] for r in state.db.execute("SELECT run_id FROM runs ORDER BY run_id")]
        for i, rid in enumerate(all_ids, 1):
            p = state.db.execute("SELECT * FROM profiles WHERE run_id=?", (rid,)).fetchone()
            p = dict(p) if p else {"status": "not_attempted", "raw_file": "", "sha256": "",
                "fetched_at": "", "error": "", "records": 0, "flags": "[]"}
            status, err = p["status"], p["error"]
            flags = set(loads(p["flags"]))
            species, obj = [], {}
            if status not in {"not_attempted", "failed", "schema_error", "cache_error"}:
                try:
                    with gzip.open(root / p["raw_file"], "rb") as f:
                        raw = f.read()
                    if digest(raw) != p["sha256"]:
                        raise ValueError("Cached response checksum differs from checkpoint")
                    obj = loads(raw)
                    checked_status, _, species, more = profile_info(obj, rid)
                    if checked_status != status:
                        raise ValueError("Cached response status differs from checkpoint")
                    flags.update(more)
                except (OSError, EOFError, ValueError, UnicodeError) as exc:
                    status, err = "cache_error", str(exc)
                    state.db.execute("UPDATE profiles SET status=?,error=? WHERE run_id=?", (status, err, rid))
                    LOG.warning("%s: %s", rid, err)
            genus_rows = obj.get("genus") if isinstance(obj.get("genus"), list) else []
            genus_ids = {cell(x.get("ncbi_taxon_id")) for x in genus_rows
                         if isinstance(x, dict) and cell(x.get("ncbi_taxon_id")).isdigit()
                         and int(cell(x.get("ncbi_taxon_id"))) > 0}
            counts = collections.Counter(cell(x.get("ncbi_taxon_id")) for x in species
                                         if isinstance(x, dict) and x.get("ncbi_taxon_id") is not None)
            per_taxon = collections.Counter()
            numeric_sum = decimal.Decimal(0)
            bad_numeric = collisions = duplicates = exported = 0
            for j, record in enumerate(species, 1):
                rf = []
                if not isinstance(record, dict):
                    record = {}
                    rf.append("non_object_record_preserved_in_raw_response")
                taxid = cell(record.get("ncbi_taxon_id", record.get("taxon_id")))
                name = cell(record.get("scientific_name", record.get("name")))
                rank = cell(record.get("taxon_rank_level", record.get("rank")))
                value = record.get("relative_abundance", record.get("abundance"))
                numeric, number = abundance_status(value)
                if number is not None:
                    numeric_sum += number
                if numeric != "valid":
                    bad_numeric += 1
                    rf.append("abundance_" + numeric)
                collision = bool(taxid and taxid in genus_ids)
                duplicate = bool(taxid and counts[taxid] > 1)
                collisions += int(collision)
                duplicates += int(duplicate)
                if not taxid or not taxid.isdigit() or int(taxid) <= 0:
                    rf.append("unresolved_or_placeholder_taxon_id")
                if rank.casefold() != "species":
                    rf.append("reported_rank_not_species")
                if collision:
                    rf.append("taxon_id_also_in_genus_table")
                if duplicate:
                    rf.append("duplicate_taxon_id_records_retained")
                flags.update(rf)
                sw.writerow([rid, j, taxid, name, rank, cell(value), numeric,
                    int(collision), int(duplicate), dumps(rf), p["raw_file"], p["sha256"]])
                key = (taxid, name, rank)
                per_taxon[key] += 1
                t = taxa.setdefault(key, {"records": 0, "runs": 0, "genus": False})
                t["records"] += 1
                t["genus"] = t["genus"] or collision
                exported += 1
            for key in per_taxon:
                taxa[key]["runs"] += 1
            if status == "downloaded" and flags:
                displayed_status = "downloaded_with_flags"
            else:
                displayed_status = status
            observed, meshes, terms, qc = merged_metadata(state, rid)
            sources = [r[0] for r in state.db.execute(
                "SELECT source FROM memberships WHERE run_id=? ORDER BY source", (rid,))]
            conflicts = sorted(k for k, v in observed.items() if len(v) > 1)
            def merged_cell(k):
                vals = observed.get(k, [])
                return cell(vals[0]) if len(vals) == 1 else (dumps(vals) if vals else "")
            mw.writerow([rid, dumps(meshes), dumps(terms), dumps(qc),
                "yes" if ASD in meshes else "no_ASD_label_observed",
                dumps(sources), displayed_status, dumps(conflicts), dumps(observed)] +
                [merged_cell(k) for k in extras])
            aw.writerow([rid, displayed_status, p["records"], exported,
                str(numeric_sum) if species else "", bad_numeric, collisions,
                duplicates, dumps(sorted(flags)), p["raw_file"], p["sha256"], p["fetched_at"], err])
            status_counts[displayed_status] += 1
            totals["runs"] += 1
            totals["species_records"] += exported
            totals["ASD_labelled_runs_retained"] += int(ASD in meshes)
            totals["runs_with_numeric_flags"] += int(bad_numeric > 0)
            totals["runs_with_genus_id_collisions"] += int(collisions > 0)
            if i % 5000 == 0:
                LOG.info("Exported %s/%s runs", f"{i:,}", f"{len(all_ids):,}")
    with (staging / names[2]).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["catalogue_record_id", "ncbi_taxon_id", "scientific_name", "reported_rank",
                    "records", "runs", "also_reported_in_genus_table", "taxonomy_validation"])
        for key in sorted(taxa):
            t = taxa[key]
            w.writerow([digest(dumps(key).encode())[:20], *key, t["records"], t["runs"],
                        int(t["genus"]), "not_externally_validated"])
    hashes = {}
    for name in names:
        h = hashlib.sha256()
        with (staging / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        hashes[name] = {"sha256": h.hexdigest(), "bytes": (staging / name).stat().st_size}
    for name in names:
        os.replace(staging / name, root / name)
    with contextlib.suppress(OSError):
        staging.rmdir()
    bad_statuses = {"not_attempted", "failed", "schema_error", "cache_error"}
    unresolved = sum(status_counts[k] for k in bad_statuses)
    inventory_ok = state.get("inventory_finished", False)
    complete = inventory_ok and not unresolved and not args.inventory_only and not interrupted
    if args.max_runs and args.max_runs < totals["runs"]:
        complete = False
    failed_calls = [dict(r) for r in state.db.execute(
        "SELECT endpoint,payload,status,error FROM calls WHERE status!='ok' ORDER BY endpoint,payload")]
    summary = {"script_version": VERSION, "created_at": state.get("created_at"),
        "updated_at": utc(), "status": "request_extraction_complete" if complete else "incomplete_or_inventory_only",
        "extraction_requests_complete": complete, "inventory_checks_passed": inventory_ok,
        "unresolved_profile_requests": unresolved, "counts": dict(totals),
        "profile_status_counts": dict(status_counts), "catalogue_records": len(taxa),
        "inventory_audit": state.get("inventory_audit", {}), "unresolved_api_calls": failed_calls,
        "files": hashes, "biological_filters_applied": [], "abundances": "as_reported; no rounding or normalization",
        "taxonomy": "reported species entries retained; external rank validation deferred",
        "important": ["Empty/unavailable profiles are not zero-filled samples.",
          "No ASD, QC, prevalence, abundance or richness filtering is applied.",
          "No run-by-species model matrix is created at extraction time.",
          "Completeness refers to the checked API inventories, not inaccessible or unpublished data.",
          "Metadata requests may retrieve amplicon rows; only metagenomic inventory members are exported."]}
    atomic_json(root / "extraction_summary.json", summary)
    state.db.commit()
    LOG.info("Wrote %s runs and %s source species records", f"{totals['runs']:,}", f"{totals['species_records']:,}")
    LOG.info("Status: %s. Full details: %s", summary["status"], root / "extraction_summary.json")
    return 0 if complete or (args.inventory_only and inventory_ok) else 2


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("metagenomic_totla"), help="Output and resume directory")
    p.add_argument("--workers", type=int, default=10, help="Concurrent I/O workers; no GPU needed")
    p.add_argument("--requests-per-second", type=float, default=2.0, help="Global request rate across all workers")
    p.add_argument("--page-size", type=int, default=1000, help="Metadata records per request")
    p.add_argument("--retries", type=int, default=6, help="Attempts per uncached request")
    p.add_argument("--timeout", type=float, default=60, help="Network socket timeout in seconds")
    p.add_argument("--max-response-mb", type=int, default=256, help="Maximum JSON response size")
    p.add_argument("--max-pages", type=int, default=10000, help="Safeguard for expanding metadata pagination")
    p.add_argument("--max-consecutive-errors", type=int, default=25, help="Pause downloads during sustained API failures")
    p.add_argument("--retry-empty", action="store_true", help="Re-query cached empty/not-found profiles; preserve previous responses")
    p.add_argument("--inventory-only", action="store_true", help="Collect metadata and inventory, without profile downloads")
    p.add_argument("--export-only", action="store_true", help="Rebuild tables from checkpoint and raw files, without network calls")
    p.add_argument("--max-runs", type=int, default=0, help="Diagnostic profile limit; 0 means all inventoried runs")
    p.add_argument("--base-url", default="https://gmrepo.humangut.info/api/", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not (1 <= a.workers <= 32) or not (0 < a.requests_per_second <= 10):
        p.error("Use 1-32 workers and a request rate greater than 0 and at most 10/s")
    if not (1 <= a.page_size <= 1000) or a.retries < 1 or a.timeout <= 0:
        p.error("Page size must be 1-1000; retries and timeout must be positive")
    if a.max_runs < 0 or a.max_response_mb < 1 or a.max_pages < 1 or a.max_consecutive_errors < 1:
        p.error("Invalid numeric safeguard")
    parsed = urllib.parse.urlparse(a.base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        p.error("A remote API must use HTTPS")
    return a


def main(argv=None):
    args = arguments(argv)
    STOP.clear()
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required")
    root = args.out.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(root / "extraction.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    lock_file = (root / "extraction.lock").open("a+")
    try:
        import fcntl
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        raise SystemExit("Run this extractor on Linux/macOS (including Linux GPU servers)")
    except (OSError, BlockingIOError):
        raise SystemExit("Cannot lock this output directory. Another extractor may be running; use one process per directory.")
    api = API(root, args)
    state = State(root, api.base)
    script = Path(__file__).resolve()
    if script.exists() and script != root / script.name:
        destination = root / script.name
        if destination.exists() and destination.read_bytes() != script.read_bytes():
            shutil.copy2(destination, root / (destination.stem + ".previous.py"))
        shutil.copy2(script, destination)
    LOG.info("GMrepo species extractor %s; output: %s", VERSION, root)
    LOG.info("%s workers; %.2f network requests/s globally; ASD and QC records retained",
             args.workers, args.requests_per_second)
    interrupted, fatal = False, ""
    try:
        if args.export_only:
            if not state.db.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
                raise ValueError("No inventoried runs found for --export-only")
        else:
            inventory(api, state, args)
            if not args.inventory_only:
                download_profiles(api, state, args)
    except KeyboardInterrupt:
        STOP.set()
        interrupted = True
        LOG.warning("Interrupted. Preserving completed downloads and exporting current state.")
    except Exception as exc:
        fatal = type(exc).__name__ + ": " + str(exc)
        LOG.exception("Extraction did not finish; cached responses and checkpoint are preserved")
        state.set("last_error", {"at": utc(), "error": fatal})
        state.db.commit()
    try:
        result = export_files(state, args, interrupted=interrupted or bool(fatal))
        return 130 if interrupted else (2 if fatal or STOP.is_set() else result)
    finally:
        state.db.close()
        lock_file.close()


if __name__ == "__main__":
    sys.exit(main())
