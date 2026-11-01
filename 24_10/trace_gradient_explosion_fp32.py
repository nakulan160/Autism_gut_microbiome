#!/usr/bin/env python3
"""Replay and trace the original GMrepo gradient explosion, entirely in float32.

ONE standalone file. The original trainer is embedded verbatim and SHA-256 checked.
Requires the same PyTorch/CUDA environment as the failing run and NumPy.

Example, from the GPU project folder:
  python -u trace_gradient_explosion_fp32.py \
    --checkpoint /mnt/data/shyam/nakulan/22_10/Encoder_only/results/checkpoints/epoch_0005.pt \
    --input non_asd_qc_pass_reported_required.tsv.gz \
    --device cuda --output-dir attention_diagnostic_fp32

Only three completed outputs: diagnostic_bundle.pt, operation_trace.tsv, report.txt.
The checkpoint supplies settings and data/vocabulary identity, never damaged state.
Starts at epoch 1, seed from the original checkpoint; stops on nonfinite state or
at update 4,113 by default. All original model/loss/Adam arithmetic remains FP32.
No clipping, no changed learning rate, no new normalization, no float64 comparison,
no alternative attention backend, no new dropout/mask choices for detailed tracing.

Observation changes: forward hooks retain module tensors, .backward retains its
native forward graph, and gradients before each sample are copied. The watched
sample's FIRST backward is directly traced, including individual parameter gradients.
A second backward uses that SAME forward graph (including saved dropout). Unexpected
spikes are traced on the retained graph. Original gradients and RNG are restored, checked,
and the original group and Adam step continue. Returning None from all node hooks
leaves the computed gradients unchanged. Exact GPU repetition is still measured,
not assumed. Temporary atomic-save files are removed after a successful save.

The tensor bundle includes complete pre-Adam state, original and isolated gradients,
all readable saved autograd operands, incoming/outgoing node gradients, data/masks,
provenance, and detached FP32 algebra checks. Native fused attention is one exposed
node: no claim is made to identify hidden individual GPU instructions from hooks.
The report distinguishes original observations from posthoc reconstructed scores.

Only load checkpoints/bundles that you created or otherwise trust.
Official API references consulted (PyTorch 2.8):
https://docs.pytorch.org/docs/2.8/generated/torch.autograd.graph.Node.register_hook.html
https://docs.pytorch.org/docs/2.8/generated/torch.autograd.graph.Node.register_prehook.html
https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html
https://docs.pytorch.org/docs/2.8/notes/randomness.html
"""

ORIGINAL_TRAINER_SHA256 = '8d3d6823fab0ff364ceebc2aa7cdd58f547042cc59b29468f554a64300374c9a'
ORIGINAL_TRAINER_SOURCE = r'''#!/usr/bin/env python3
"""Train an abundance-only, CLS Transformer on reported GMrepo profiles.

All input runs are training runs. Blank cells are unreported, never zero-filled.
Only the exact label 'Unknown' is excluded by default. Species IDs are external
bookkeeping: ID s selects output index s-1. They never enter the encoder.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import torch
from torch import nn


# Change defaults here, use --config settings.json, or use the CLI overrides.
@dataclass
class Config:
    input: str = "metagenomic_totla/non_asd_qc_pass_reported_blank.tsv.gz"
    output_dir: str = "metagenomic_totla/transformer_no_unknown"
    excluded_features: tuple[str, ...] = ("Unknown",)
    epochs: int = 100
    mask_fraction: float = 0.10
    accumulate_samples: int = 16
    seed: int = 42
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    encoder_dropout: float = 0.10
    head_hidden: tuple[int, ...] = (1024, 2048)
    scale_abundance_embedding: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-9
    warmup_updates: int = 4000
    learning_rate_factor: float = 1.0
    checkpoint_every: int = 5
    plot_every: int = 10
    diagnostic_samples: int = 128
    save_mask_logs: bool = True
    progress_every: int = 1000
    device: str = "auto"
    threads: int = 10

    def validate(self) -> None:
        positive = ("epochs", "accumulate_samples", "d_model", "num_heads",
                    "num_layers", "d_ff", "warmup_updates", "checkpoint_every",
                    "plot_every", "diagnostic_samples", "progress_every", "threads")
        for key in positive:
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask_fraction must be between zero and one.")
        if not 0 <= self.encoder_dropout < 1:
            raise ValueError("encoder_dropout must be in [0, 1).")
        if not all(0 <= b < 1 for b in (self.adam_beta1, self.adam_beta2)):
            raise ValueError("Adam betas must be in [0, 1).")
        if self.adam_epsilon <= 0 or self.learning_rate_factor <= 0:
            raise ValueError("Adam epsilon and learning_rate_factor must be positive.")
        if any(not isinstance(w, int) or w < 1 for w in self.head_hidden):
            raise ValueError("Every head_hidden width must be a positive integer.")


@dataclass(frozen=True)
class Vocabulary:
    labels: tuple[str, ...]
    source_columns: tuple[int, ...]  # Positions in the original TSV, including run_id.
    source_header: tuple[str, ...]
    removed_labels: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.labels, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Profile:
    run_id: str
    species_ids: np.ndarray
    abundances: np.ndarray
    abundance_text: tuple[str, ...]


@dataclass(frozen=True)
class MaskedProfile:
    run_id: str
    binary_mask: np.ndarray
    visible_species_ids: np.ndarray
    visible_abundances: np.ndarray
    masked_species_ids: np.ndarray
    targets: np.ndarray
    target_text: tuple[str, ...]


def open_text(path: str | Path, mode: str = "rt") -> TextIO:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8", newline="")


@contextmanager
def atomic_text(path: Path) -> Iterator[TextIO]:
    """An interrupted write never replaces an existing completed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    tmp = Path(temporary)
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(tmp, "wt", encoding="utf-8", newline="") as handle:
            yield handle
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    with atomic_text(path) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


# 1. Vocabulary: one permanent output position per retained original column.
def build_vocabulary(input_path: str | Path,
                     excluded_features: tuple[str, ...] = ("Unknown",)) -> Vocabulary:
    with open_text(input_path) as handle:
        header = next(csv.reader(handle, delimiter="\t"), None)
    if not header or header[0] != "run_id":
        raise ValueError("Expected a TSV whose first column is exactly run_id.")
    if any(not name for name in header[1:]) or len(set(header)) != len(header):
        raise ValueError("Feature column labels must be nonempty and unique.")
    excluded = set(excluded_features)
    kept = tuple(i for i in range(1, len(header)) if header[i] not in excluded)
    if not kept:
        raise ValueError("No features remain after the requested exclusions.")
    return Vocabulary(tuple(header[i] for i in kept), kept, tuple(header),
                      tuple(name for name in header[1:] if name in excluded))


# 2. Load only reported cells; optionally export the same rows without Unknown.
def load_profiles(input_path: str | Path, vocabulary: Vocabulary,
                  filtered_output: Path | None = None) -> tuple[list[Profile], dict]:
    profiles: list[Profile] = []
    seen: set[str] = set()
    fingerprint = hashlib.sha256(vocabulary.sha256.encode())
    counts: list[int] = []
    populated_before = removed_cells = reported_zeros = 0
    kept_columns = set(vocabulary.source_columns)
    removed_columns = [i for i in range(1, len(vocabulary.source_header)) if i not in kept_columns]
    if filtered_output is not None and Path(input_path).resolve() == filtered_output.resolve():
        if vocabulary.removed_labels:
            raise ValueError("Filtered output must not overwrite the input file.")
        filtered_output = None  # Already-filtered input in its own output folder.
    output_context = atomic_text(filtered_output) if filtered_output else nullcontext(None)
    with open_text(input_path) as source, output_context as destination:
        reader = csv.reader(source, delimiter="\t")
        if tuple(next(reader)) != vocabulary.source_header:
            raise ValueError("Input header changed after the vocabulary was built.")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n") if destination else None
        if writer:
            writer.writerow(("run_id", *vocabulary.labels))
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(vocabulary.source_header):
                raise ValueError(f"Line {line_number}: inconsistent number of TSV columns.")
            run_id = row[0]
            if not run_id or run_id in seen:
                raise ValueError(f"Line {line_number}: empty or repeated run_id {run_id!r}.")
            seen.add(run_id)
            retained_text = [row[column] for column in vocabulary.source_columns]
            removed_count = sum(row[column] != "" for column in removed_columns)
            removed_cells += removed_count
            ids: list[int] = []
            values: list[float] = []
            original_text: list[str] = []
            for species_id, text in enumerate(retained_text, 1):
                if text == "":
                    continue  # Blank is unreported; no input or target is created.
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(f"{run_id}, ID {species_id}: invalid abundance {text!r}.") from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{run_id}, ID {species_id}: abundance must be finite and nonnegative.")
                value32 = np.float32(value)
                if not np.isfinite(value32) or (value != 0 and value32 == 0):
                    raise ValueError(f"{run_id}, ID {species_id}: abundance exceeds float32 range.")
                ids.append(species_id)
                values.append(value)
                original_text.append(text)
                reported_zeros += int(value == 0)
            if writer:
                writer.writerow((run_id, *retained_text))  # Strings unchanged, blanks unchanged.
            counts.append(len(ids))
            populated_before += len(ids) + removed_count
            profiles.append(Profile(run_id, np.asarray(ids, dtype=np.int64),
                                    np.asarray(values, dtype=np.float32), tuple(original_text)))
            canonical = json.dumps([run_id, ids, original_text], ensure_ascii=False,
                                   separators=(",", ":")).encode()
            fingerprint.update(len(canonical).to_bytes(8, "big"))
            fingerprint.update(canonical)
            if len(profiles) % 5000 == 0:
                print(f"Loaded {len(profiles):,} runs", flush=True)
    if not profiles:
        raise ValueError("The input contains no sample rows.")
    return profiles, {
        "input": str(Path(input_path).resolve()), "runs": len(profiles),
        "input_feature_columns": len(vocabulary.source_header) - 1,
        "retained_feature_columns": vocabulary.size,
        "removed_labels": list(vocabulary.removed_labels),
        "populated_cells_before": populated_before, "removed_reported_cells": removed_cells,
        "populated_cells_after": sum(counts), "reported_zero_cells_retained": reported_zeros,
        "minimum_reported_entries": min(counts), "maximum_reported_entries": max(counts),
        "single_entry_runs": [p.run_id for p in profiles if len(p.species_ids) == 1],
        "empty_runs": [p.run_id for p in profiles if len(p.species_ids) == 0],
        "data_sha256": fingerprint.hexdigest(), "vocabulary_sha256": vocabulary.sha256,
        "missing_values": "blank; never imputed", "training_dtype": "float32",
        "abundance_transformation": "none", "all_runs_are_training_runs": True,
    }


# 3. Save targets, multiply by the binary mask, then remove masked entries.
def mask_profile(profile: Profile, mask_fraction: float, rng: np.random.Generator | None,
                 masked_species_ids: list[int] | None = None) -> MaskedProfile:
    n = len(profile.species_ids)
    if n == 0:
        raise ValueError(f"{profile.run_id}: no reported entries remain to supply a target.")
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between zero and one.")
    if masked_species_ids is None:
        if rng is None:
            raise ValueError("A random generator is required for a new mask.")
        k = max(1, math.floor(mask_fraction * n))
        positions = np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
    else:
        if not masked_species_ids or len(set(masked_species_ids)) != len(masked_species_ids):
            raise ValueError("Saved diagnostic masked IDs must be nonempty and unique.")
        lookup = {int(species_id): i for i, species_id in enumerate(profile.species_ids)}
        try:
            positions = np.asarray([lookup[s] for s in masked_species_ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{profile.run_id}: saved masked ID was not originally reported.") from exc
    targets = profile.abundances[positions].copy()
    target_ids = profile.species_ids[positions].copy()
    target_text = tuple(profile.abundance_text[int(i)] for i in positions)
    binary_mask = np.ones(n, dtype=np.bool_)
    binary_mask[positions] = False
    corrupted = profile.abundances * binary_mask.astype(np.float32)
    # Index by the binary mask, not corrupted != 0: genuine reported zeros stay visible.
    return MaskedProfile(profile.run_id, binary_mask, profile.species_ids[binary_mask].copy(),
                         corrupted[binary_mask].copy(), target_ids, targets, target_text)


# 4. One variable-length sample at a time: no padding and no species input embedding.
def prepare_batch(masked: MaskedProfile, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = torch.from_numpy(masked.visible_abundances).to(device).reshape(1, -1, 1)
    ids = torch.from_numpy(masked.masked_species_ids).to(device)
    targets = torch.from_numpy(masked.targets).to(device)
    return values, ids, targets


class AbundanceEmbedding(nn.Module):
    """Project each scalar abundance to d_model; no input dropout or position signal."""
    def __init__(self, d_model: int, scale: bool):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, abundance: torch.Tensor) -> torch.Tensor:
        return self.projection(abundance) * self.scale


class PredictionHead(nn.Module):
    """CLS 512 -> 1024 -> ReLU -> 2048 -> ReLU -> V, with linear final output."""
    def __init__(self, d_model: int, hidden: tuple[int, ...], vocabulary_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = d_model
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, vocabulary_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.layers(cls)


class CLSEncoder(nn.Module):
    def __init__(self, config: Config, vocabulary_size: int):
        super().__init__()
        self.embedding = AbundanceEmbedding(config.d_model, config.scale_abundance_embedding)
        # One shared learned CLS vector; initialized once, then optimized with the model.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        # Construct each block separately so initial parameter values are independent.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.num_heads,
                                       dim_feedforward=config.d_ff, dropout=config.encoder_dropout,
                                       activation="relu", batch_first=True, norm_first=False)
            for _ in range(config.num_layers)
        ])
        self.head = PredictionHead(config.d_model, config.head_hidden, vocabulary_size)

    def encode_sample(self, abundances: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(abundances)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)  # CLS is always the first token.
        for block in self.blocks:
            tokens = block(tokens)  # Bidirectional self-attention; no causal mask.
        return tokens[:, 0, :]

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_sample(abundances))


# 5. Model construction is isolated from the training loop.
def build_model(config: Config, vocabulary_size: int, device: torch.device) -> CLSEncoder:
    return CLSEncoder(config, vocabulary_size).to(device)


# 6. Exact identity matching: species ID s -> output index s-1, in every sample.
def compute_masked_loss(predictions: torch.Tensor, masked_species_ids: torch.Tensor,
                        targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError("Expected the full prediction vector for exactly one sample.")
    if masked_species_ids.ndim != 1 or targets.shape != masked_species_ids.shape or targets.numel() == 0:
        raise ValueError("Masked IDs and targets must have identical, nonempty 1D shapes.")
    if masked_species_ids.dtype != torch.long:
        raise ValueError("Species IDs must be integer torch.long values.")
    if bool(((masked_species_ids < 1) | (masked_species_ids > predictions.shape[1])).any()):
        raise ValueError("Masked species ID lies outside the saved vocabulary.")
    if masked_species_ids.unique().numel() != masked_species_ids.numel():
        raise ValueError("A species may appear at most once in a sample's targets.")
    selected = predictions[0].index_select(0, masked_species_ids - 1)
    return (selected - targets).square().mean(), selected


def transformer_learning_rate(update: int, config: Config) -> float:
    if update < 1:
        raise ValueError("Optimizer updates start at one.")
    return (config.learning_rate_factor * config.d_model ** -0.5
            * min(update ** -0.5, update * config.warmup_updates ** -1.5))


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Adam:
    # LR is set from the actual optimizer update number immediately before step().
    return torch.optim.Adam(model.parameters(), lr=0.0,
                            betas=(config.adam_beta1, config.adam_beta2),
                            eps=config.adam_epsilon, weight_decay=0.0)


# 7. Per-sample masked MSE, then equal weighting of samples in each update/epoch.
def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    profiles: list[Profile], config: Config, rng: np.random.Generator,
                    device: torch.device, epoch: int, global_step: int) -> tuple[dict, int]:
    model.train()
    started = time.perf_counter()
    order = rng.permutation(len(profiles))
    mse_sum = mae_sum = 0.0
    masked_entries = seen = cls_only = 0
    next_progress = config.progress_every
    path = Path(config.output_dir) / "masked_targets" / f"epoch_{epoch:04d}.tsv.gz"
    log_context = atomic_text(path) if config.save_mask_logs else nullcontext(None)
    with log_context as log:
        writer = csv.writer(log, delimiter="\t", lineterminator="\n") if log else None
        if writer:
            writer.writerow(("run_id", "species_id", "output_index", "original_abundance", "prediction"))
        for start in range(0, len(order), config.accumulate_samples):
            group = order[start:start + config.accumulate_samples]
            optimizer.zero_grad(set_to_none=True)
            for profile_index in group:
                masked = mask_profile(profiles[int(profile_index)], config.mask_fraction, rng)
                values, ids, targets = prepare_batch(masked, device)
                loss, selected = compute_masked_loss(model(values), ids, targets)
                sample_mse = float(loss.detach())
                if not math.isfinite(sample_mse):
                    raise FloatingPointError(f"Nonfinite loss: epoch {epoch}, run {masked.run_id}.")
                # The last partial group uses its actual size, never a fixed divisor of 16.
                (loss / len(group)).backward()
                mse_sum += sample_mse
                mae_sum += float((selected.detach() - targets).abs().mean())
                seen += 1
                masked_entries += len(masked.masked_species_ids)
                cls_only += int(len(masked.visible_abundances) == 0)
                if writer:
                    predicted = selected.detach().cpu().numpy()
                    for species_id, text, prediction in zip(masked.masked_species_ids,
                                                            masked.target_text, predicted):
                        writer.writerow((masked.run_id, int(species_id), int(species_id) - 1,
                                         text, format(float(prediction), ".9g")))
            global_step += 1
            learning_rate = transformer_learning_rate(global_step, config)
            for group_parameters in optimizer.param_groups:
                group_parameters["lr"] = learning_rate
            # No gradient clipping, as requested.
            optimizer.step()
            if seen >= next_progress or seen == len(profiles):
                print(f"Epoch {epoch:03d}: {seen:,}/{len(profiles):,} runs; "
                      f"mean sample MSE={mse_sum / seen:.8g}; lr={learning_rate:.6g}", flush=True)
                next_progress = seen + config.progress_every
    epoch_mse = mse_sum / len(profiles)
    return {
        "epoch": epoch, "train_mse": epoch_mse, "train_rmse": math.sqrt(epoch_mse),
        "train_mae": mae_sum / len(profiles), "samples": len(profiles),
        "masked_entries": masked_entries, "cls_only_samples": cls_only,
        "optimizer_updates_total": global_step, "learning_rate_last": learning_rate,
        "seconds": time.perf_counter() - started,
    }, global_step


def make_diagnostic_plan(profiles: list[Profile], config: Config) -> list[dict]:
    # Separate RNG: plotting choices do not alter the training shuffle/mask stream.
    rng = np.random.default_rng(config.seed + 1)
    indices = np.sort(rng.choice(len(profiles), min(config.diagnostic_samples, len(profiles)),
                                 replace=False))
    plan = []
    for index in indices:
        profile = profiles[int(index)]
        masked = mask_profile(profile, config.mask_fraction, rng)
        plan.append({"profile_index": int(index), "run_id": profile.run_id,
                     "masked_species_ids": masked.masked_species_ids.tolist()})
    return plan


# 8. Fixed training examples and masks; evaluated with dropout disabled.
def collect_training_diagnostics(model: nn.Module, profiles: list[Profile], vocabulary: Vocabulary,
                                 config: Config, plan: list[dict], device: torch.device) -> list[dict]:
    records = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for item in plan:
                profile = profiles[item["profile_index"]]
                if profile.run_id != item["run_id"]:
                    raise ValueError("Diagnostic run order no longer matches the saved plan.")
                masked = mask_profile(profile, config.mask_fraction, None, item["masked_species_ids"])
                values, ids, targets = prepare_batch(masked, device)
                _, selected = compute_masked_loss(model(values), ids, targets)
                predicted = selected.cpu().numpy()
                for species_id, text, target, prediction in zip(masked.masked_species_ids,
                                                                 masked.target_text,
                                                                 masked.targets, predicted):
                    prediction_value = float(prediction)
                    target_value = float(target)
                    if not math.isfinite(prediction_value):
                        raise FloatingPointError("Nonfinite diagnostic prediction.")
                    records.append({"run_id": profile.run_id, "species_id": int(species_id),
                                    "output_index": int(species_id) - 1,
                                    "reported_label": vocabulary.labels[int(species_id) - 1],
                                    "original_abundance": text, "target_float32": target_value,
                                    "prediction": prediction_value,
                                    "residual": prediction_value - target_value})
    finally:
        model.train(was_training)
    return records


# Settings that must agree when resuming the same training trajectory.
RESUME_KEYS = ("mask_fraction", "accumulate_samples", "seed", "d_model", "num_heads",
               "num_layers", "d_ff", "encoder_dropout", "head_hidden",
               "scale_abundance_embedding", "adam_beta1", "adam_beta2", "adam_epsilon",
               "warmup_updates", "learning_rate_factor", "diagnostic_samples")


# 9. Full resumable checkpoint at completed epochs 5, 10, 15, ... by default.
def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    config: Config, vocabulary: Vocabulary, data_sha256: str,
                    epoch: int, global_step: int, history: list[dict],
                    rng: np.random.Generator, diagnostic_plan: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "epoch": epoch, "global_step": global_step,
        "config": asdict(config), "vocabulary_labels": vocabulary.labels,
        "vocabulary_sha256": vocabulary.sha256, "data_sha256": data_sha256,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "lr_schedule": {"name": "transformer_warmup_inverse_sqrt", "update": global_step,
                        "last_lr": optimizer.param_groups[0]["lr"]},
        "history": history, "diagnostic_plan": diagnostic_plan,
        "numpy_generator_state": rng.bit_generator.state, "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "torch_version": str(torch.__version__),
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


# 10. Check vocabulary identity, data, settings, optimizer, and RNG state.
def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    config: Config, vocabulary: Vocabulary, data_sha256: str,
                    rng: np.random.Generator) -> dict:
    # These are this program's own full checkpoints; use only a trusted checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    if (checkpoint["vocabulary_sha256"] != vocabulary.sha256
            or tuple(checkpoint["vocabulary_labels"]) != vocabulary.labels):
        raise ValueError("Checkpoint species identities/order differ from the input vocabulary.")
    if checkpoint["data_sha256"] != data_sha256:
        raise ValueError("Checkpoint and input do not contain identical retained profiles.")
    changed = [key for key in RESUME_KEYS
               if checkpoint["config"][key] != asdict(config)[key]]
    if changed:
        raise ValueError("Resume settings differ: " + ", ".join(changed))
    if checkpoint["epoch"] > config.epochs:
        raise ValueError("Requested epochs is earlier than the checkpoint epoch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng.bit_generator.state = checkpoint["numpy_generator_state"]
    random.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_states"]:
        if len(checkpoint["cuda_rng_states"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        else:
            print("CUDA device count changed; continuation is not bitwise reproducible.", flush=True)
    return checkpoint


# 11. Metrics every epoch; vocabulary, effective settings, and audit also saved.
def save_results(output_dir: Path, history: list[dict], diagnostic_records: list[dict] | None = None,
                 epoch: int | None = None) -> None:
    if history:
        with atomic_text(output_dir / "epoch_metrics.csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    if diagnostic_records is not None:
        if not diagnostic_records or epoch is None:
            raise ValueError("Diagnostic records and their epoch are required together.")
        path = output_dir / "diagnostics" / f"epoch_{epoch:04d}.tsv.gz"
        with atomic_text(path) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_records[0]),
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(diagnostic_records)


def save_preparation_files(output_dir: Path, config: Config, vocabulary: Vocabulary, audit: dict) -> None:
    write_json(output_dir / "settings_used.json", asdict(config))
    write_json(output_dir / "input_audit.json", audit)
    write_json(output_dir / "vocabulary.json", {
        "vocabulary_sha256": vocabulary.sha256, "vocabulary_size": vocabulary.size,
        "cls_id": 0, "cls_is_input_only": True, "output_index_rule": "species_id - 1",
        "excluded_labels": list(config.excluded_features),
        "features": [{"species_id": i, "output_index": i - 1, "reported_label": name}
                     for i, name in enumerate(vocabulary.labels, 1)],
    })
    with atomic_text(output_dir / "species_vocabulary.tsv") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("species_id", "output_index", "reported_label"))
        writer.writerows((i, i - 1, name) for i, name in enumerate(vocabulary.labels, 1))


def plotting_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# 12. Plot all stored epoch metrics through the current plotting epoch.
def plot_training_loss(history: list[dict], path: Path) -> None:
    plt = plotting_module()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([row["epoch"] for row in history], [row["train_mse"] for row in history],
            color="#245e8e", linewidth=1.8)
    ax.set(xlabel="Epoch", ylabel="Mean per-sample masked MSE",
           title="Training loss (all training runs)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# 13. Original vs predicted values, only for fixed masked training targets.
def plot_predictions(records: list[dict], path: Path, epoch: int) -> None:
    plt = plotting_module()
    original = np.asarray([r["target_float32"] for r in records])
    predicted = np.asarray([r["prediction"] for r in records])
    lo, hi = float(min(original.min(), predicted.min())), float(max(original.max(), predicted.max()))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(original, predicted, s=12, alpha=0.45, color="#245e8e", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "--", color="#a34343", linewidth=1.2, label="Perfect prediction")
    ax.set(xlabel="Original masked abundance", ylabel="Predicted abundance",
           title=f"Training reconstruction: epoch {epoch}")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# 14. Residual = predicted - original; zero is the reference line.
def plot_residuals(records: list[dict], path: Path, epoch: int) -> None:
    plt = plotting_module()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([r["prediction"] for r in records], [r["residual"] for r in records],
               s=12, alpha=0.45, color="#38735a", edgecolors="none")
    ax.axhline(0, linestyle="--", color="#a34343", linewidth=1.2)
    ax.set(xlabel="Predicted abundance", ylabel="Residual (prediction - original)",
           title=f"Training reconstruction residuals: epoch {epoch}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def archive_uncheckpointed_outputs(output_dir: Path, completed_epoch: int) -> None:
    """Keep interrupted-epoch outputs separately before replaying from a checkpoint."""
    stale = []
    for subdirectory in ("masked_targets", "diagnostics", "plots"):
        folder = output_dir / subdirectory
        if not folder.exists():
            continue
        for path in folder.glob("epoch_*"):
            try:
                epoch = int(path.name.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if epoch > completed_epoch:
                stale.append(path)
    if stale:
        archive = output_dir / "interrupted_outputs" / str(time.time_ns())
        for path in stale:
            destination = archive / path.relative_to(output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)


# 15. Orchestration: all samples train; checkpoints and plots have separate schedules.
def run_training(config: Config, resume: str | None = None, prepare_only: bool = False) -> dict:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted((output_dir / "checkpoints").glob("epoch_*.pt"))
    if resume == "latest":
        if not checkpoint_paths:
            raise ValueError("No checkpoint exists in this output directory to resume.")
        resume = str(checkpoint_paths[-1])
    if not resume and (checkpoint_paths or (output_dir / "epoch_metrics.csv").exists()):
        raise ValueError("This folder already contains training. Use --resume latest or a new --output-dir.")
    if prepare_only and resume:
        raise ValueError("Use --prepare-only or --resume, not both.")
    vocabulary = build_vocabulary(config.input, config.excluded_features)
    old_vocabulary = output_dir / "vocabulary.json"
    if old_vocabulary.exists():
        old = json.loads(old_vocabulary.read_text(encoding="utf-8"))
        if old["vocabulary_sha256"] != vocabulary.sha256:
            raise ValueError("Output folder already belongs to a different vocabulary; choose a new folder.")
    filtered = output_dir / "non_asd_qc_pass_reported_no_unknown.tsv.gz"
    # Validate resumed input without replacing a previous filtered export first.
    profiles, audit = load_profiles(config.input, vocabulary, None if resume else filtered)
    print(f"Runs: {len(profiles):,}; vocabulary: {vocabulary.size:,}; "
          f"removed labels: {list(vocabulary.removed_labels)}", flush=True)
    print(f"Single-entry runs (CLS-only when masked): {len(audit['single_entry_runs'])}", flush=True)
    if prepare_only:
        save_preparation_files(output_dir, config, vocabulary, audit)
        print(f"Prepared data and dictionary in {output_dir.resolve()}", flush=True)
        return audit
    if audit["empty_runs"]:
        raise ValueError(f"{len(audit['empty_runs'])} runs have no retained reported targets; "
                         "no runs were silently dropped. Inspect input_audit with --prepare-only.")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch installation has no available CUDA device.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This script supports CPU and CUDA devices.")
    rng = np.random.default_rng(config.seed)
    model = build_model(config, vocabulary.size, device)
    optimizer = build_optimizer(model, config)
    history: list[dict] = []
    global_step = start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(Path(resume), model, optimizer, config, vocabulary,
                                     audit["data_sha256"], rng)
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        if any(int(p.stem.split("_")[1]) > start_epoch for p in checkpoint_paths):
            raise ValueError("A newer checkpoint exists here. Use --resume latest or a new output folder.")
        history = checkpoint["history"]
        diagnostic_plan = checkpoint["diagnostic_plan"]
        archive_uncheckpointed_outputs(output_dir, start_epoch)
        print(f"Resumed epoch {start_epoch}; optimizer updates {global_step:,}", flush=True)
    else:
        diagnostic_plan = make_diagnostic_plan(profiles, config)
    save_preparation_files(output_dir, config, vocabulary, audit)
    write_json(output_dir / "diagnostic_masks.json", diagnostic_plan)
    save_results(output_dir, history)
    print(f"Device: {device}; trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("Abundance-only inputs: species identities are used only for target/output matching.", flush=True)
    print("All runs train; diagnostic samples are training samples, not a held-out split.", flush=True)
    for epoch in range(start_epoch + 1, config.epochs + 1):
        metrics, global_step = train_one_epoch(model, optimizer, profiles, config, rng,
                                              device, epoch, global_step)
        history.append(metrics)
        save_results(output_dir, history)  # Always persist every completed epoch.
        if epoch % config.checkpoint_every == 0:
            path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            save_checkpoint(path, model, optimizer, config, vocabulary, audit["data_sha256"],
                            epoch, global_step, history, rng, diagnostic_plan)
            print(f"Saved checkpoint: {path}", flush=True)
        if epoch % config.plot_every == 0:
            records = collect_training_diagnostics(model, profiles, vocabulary, config,
                                                   diagnostic_plan, device)
            save_results(output_dir, history, records, epoch)
            folder = output_dir / "plots"
            plot_training_loss(history, folder / f"epoch_{epoch:04d}_loss.png")
            plot_predictions(records, folder / f"epoch_{epoch:04d}_predictions.png", epoch)
            plot_residuals(records, folder / f"epoch_{epoch:04d}_residuals.png", epoch)
            print(f"Saved three plots for epoch {epoch}", flush=True)
    summary = {"completed_epoch": config.epochs, "vocabulary_size": vocabulary.size,
               "runs": len(profiles), "optimizer_updates": global_step,
               "latest_scheduled_checkpoint": str(sorted((output_dir / 'checkpoints').glob('epoch_*.pt'))[-1])
               if list((output_dir / 'checkpoints').glob('epoch_*.pt')) else None}
    write_json(output_dir / "training_summary.json", summary)
    print(f"Completed. Results: {output_dir.resolve()}", flush=True)
    return summary


def parse_arguments() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON settings; unknown keys are rejected.")
    parser.add_argument("--input", help="Original or already-filtered run-by-feature TSV(.gz).")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0, etc.")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mask-fraction", type=float)
    parser.add_argument("--accumulate-samples", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--plot-every", type=int)
    parser.add_argument("--diagnostic-samples", type=int)
    parser.add_argument("--resume", help="latest or a checkpoint path; only load your trusted checkpoints.")
    parser.add_argument("--prepare-only", action="store_true", help="Export filtered data/dictionary, then stop.")
    args = parser.parse_args()
    values = {}
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(Config)}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("Configuration must be a JSON object containing only Config field names.")
    for field in fields(Config):
        override = getattr(args, field.name, None)
        if override is not None:
            values[field.name] = override
    for key in ("head_hidden", "excluded_features"):
        if key in values:
            values[key] = tuple(values[key])
    return Config(**values), args


if __name__ == "__main__":
    try:
        settings, arguments = parse_arguments()
        run_training(settings, arguments.resume, arguments.prepare_only)
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Resume from the latest completed scheduled checkpoint.", file=sys.stderr)
        raise SystemExit(130)
'''

"""Diagnostic runtime appended after the verbatim original trainer source."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
import types
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def load_original_trainer():
    """Execute the embedded, byte-verified original definitions, not its CLI."""
    actual = hashlib.sha256(ORIGINAL_TRAINER_SOURCE.encode('utf-8')).hexdigest()
    if actual != ORIGINAL_TRAINER_SHA256:
        raise RuntimeError('Embedded original trainer failed its SHA-256 check.')
    module = types.ModuleType('_gmrepo_original_trainer')
    module.__file__ = '<embedded-original-train_masked_abundance.py>'
    sys.modules[module.__name__] = module
    exec(compile(ORIGINAL_TRAINER_SOURCE, module.__file__, 'exec'), module.__dict__)
    return module


training = load_original_trainer()


def tensor_maxima(named_tensors):
    """The original replay's infinity-norm scan; no squared norm overflow."""
    buckets, result = {}, []
    for name, value in named_tensors:
        if value is None:
            continue
        value = value.detach()
        if not value.numel():
            result.append((name, value, 0.0))
        else:
            buckets.setdefault((value.device, value.dtype), []).append((name, value))
    for entries in buckets.values():
        tensors = [value for _, value in entries]
        norms = (torch._foreach_norm(tensors, float('inf'))
                 if hasattr(torch, '_foreach_norm') else [x.abs().amax() for x in tensors])
        maxima = torch.stack(tuple(norms)).cpu().tolist()
        result.extend((name, value, float(maximum))
                      for (name, value), maximum in zip(entries, maxima))
    return result


def scan(named_tensors):
    values = tensor_maxima(named_tensors)
    bad = [{'name': n, **tensor_statistics(t)} for n, t, m in values if not math.isfinite(m)]
    finite = sorted(((m, n) for n, _, m in values if math.isfinite(m)), reverse=True)
    return {'nonfinite': bad, 'max_abs': finite[0][0] if finite else 0.0,
            'top': [{'name': n, 'max_abs': m} for m, n in finite[:8]]}


def optimizer_tensors(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                yield names[id(parameter)] + '/' + key, value


def tensor_statistics(tensor):
    """FP32 rescaled reductions; never convert floating tensors to another dtype.

    A raw sum of squares of finite FP32 gradients can overflow. Divide by the
    maximum before reducing, then rescale the printed Python scalar. These are
    detached measurements, not modifications to gradients or model arithmetic.
    """
    x = tensor.detach()
    finite = torch.isfinite(x)
    n = int(x.numel())
    result = {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype),
              'numel': n, 'nan': int(torch.isnan(x).sum().item()),
              'posinf': int(torch.isposinf(x).sum().item()),
              'neginf': int(torch.isneginf(x).sum().item())}
    v = x[finite]
    if not v.numel():
        result.update(max_abs=None, l2=None, mean=None, std=None, min=None, max=None)
        return result
    if not v.is_floating_point():
        result.update(max_abs=float(v.abs().max().item()), l2=None, mean=None,
                      std=None, min=float(v.min().item()), max=float(v.max().item()))
        return result
    m = float(v.abs().max().item())
    if m:
        scaled = v / m
        result.update(max_abs=m, l2=m * math.sqrt(float(scaled.square().sum().item())),
                      mean=m * float(scaled.mean().item()),
                      std=m * float(scaled.std(unbiased=False).item()))
    else:
        result.update(max_abs=0.0, l2=0.0, mean=0.0, std=0.0)
    result.update(min=float(v.min().item()), max=float(v.max().item()))
    return result


def capture_rng(generator):
    return {'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy_global': np.random.get_state(),
            'numpy_generator': copy.deepcopy(generator.bit_generator.state)}


def restore_rng(state, generator):
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy_global'])
    generator.bit_generator.state = copy.deepcopy(state['numpy_generator'])


def rng_equal(a, b):
    return (torch.equal(a['torch_cpu'], b['torch_cpu'])
            and len(a['torch_cuda']) == len(b['torch_cuda'])
            and all(torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda']))
            and repr(a['python']) == repr(b['python'])
            and repr(a['numpy_global']) == repr(b['numpy_global'])
            and repr(a['numpy_generator']) == repr(b['numpy_generator']))


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


class TensorArchive:
    """Store complete tensors and summaries, sharing exact aliases when safe."""
    def __init__(self):
        self.tensors, self.statistics, self.aliases, self.live = {}, {}, {}, []

    def add(self, tensor):
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            raise TypeError('Expected a tensor.')
        key_data = (str(tensor.device), str(tensor.dtype), tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
                    tensor._version)
        if key_data in self.aliases:
            return self.aliases[key_data]
        key = 'tensor_%06d' % len(self.tensors)
        value = tensor.detach().to(device='cpu', copy=True)
        self.tensors[key] = value
        self.statistics[key] = tensor_statistics(value)
        self.statistics[key].update(source_device=str(tensor.device),
                                    source_stride=list(tensor.stride()),
                                    source_storage_offset=int(tensor.storage_offset()),
                                    source_requires_grad=bool(tensor.requires_grad))
        self.aliases[key_data] = key
        self.live.append(tensor.detach())  # Prevent allocator pointer reuse while tracing.
        return key

    def payload(self):
        return {'tensors': self.tensors, 'statistics': self.statistics}


TRACE_COLUMNS = ['record_type', 'pass_name', 'epoch', 'update', 'sample_position', 'run_id',
                 'scope', 'node_id', 'operation', 'direction', 'slot', 'tensor_key',
                 'max_abs', 'l2', 'nan', 'posinf', 'neginf', 'details_json']


class OutputFiles:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.report = (self.folder / 'report.txt').open('w', encoding='utf-8', buffering=1)
        self.trace = (self.folder / 'operation_trace.tsv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.trace, fieldnames=TRACE_COLUMNS, delimiter='\t',
                                     lineterminator='\n')
        self.writer.writeheader()

    def row(self, context, record_type, **data):
        row = {k: context.get(k, '') for k in ['epoch', 'update', 'sample_position', 'run_id']}
        row.update(record_type=record_type)
        for k in TRACE_COLUMNS:
            if k in data:
                row[k] = data.pop(k)
        if data:
            row['details_json'] = json.dumps(data, ensure_ascii=False, default=str)
        self.writer.writerow(row)

    def save(self, payload):
        self.last_payload = payload
        self.trace.flush()
        self.report.flush()
        path = self.folder / 'diagnostic_bundle.pt'
        fd, tmp = tempfile.mkstemp(prefix='.diagnostic_bundle.', dir=self.folder)
        os.close(fd)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def close(self):
        self.trace.close()
        self.report.close()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ForwardRecorder:
    """Module forward hooks return None; they only retain this sample's tensors.

    Training mode already disables the Transformer/MHA inference fast paths.
    No full backward module hooks, new input requires_grad flags, attention
    weights requests, backend substitutions, or random draws are introduced.
    """
    def __init__(self, model):
        self.events, self.roots, self.handles = [], {}, []
        self.parameter_names = {id(p): name for name, p in model.named_parameters()}
        for name, module in model.named_modules():
            label = name or 'model'
            self.handles.append(module.register_forward_hook(self.hook(label)))

    def reset(self):
        self.events.clear()
        self.roots.clear()

    def hook(self, label):
        def observe(module, inputs, output):
            outputs = output if isinstance(output, (tuple, list)) else (output,)
            self.events.append((label, inputs, outputs))
            for value in outputs:
                if torch.is_tensor(value) and value.grad_fn is not None:
                    self.roots.setdefault(value.grad_fn, []).append(label)
            return None
        return observe

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.reset()


class GraphRecorder:
    """Observe every exposed native autograd node in the retained forward graph.

    pre-hook: gradient entering a backward operation (grad_outputs).
    post-hook: gradients returned toward that operation's forward inputs.
    All hooks explicitly return None and never replace gradients.
    """
    def __init__(self, scaled_loss, forward, archive, output, context,
                 pass_name='retained_graph_backward'):
        self.archive, self.output, self.context = archive, output, context
        self.pass_name = pass_name
        self.nodes, self.handles, self.execution = [], [], 0
        self.forward_records = []
        self.errors = []
        for label, inputs, outputs in forward.events:
            for direction, values in [('input', inputs), ('output', outputs)]:
                for slot, value in enumerate(values):
                    if torch.is_tensor(value):
                        key = archive.add(value)
                        self.forward_records.append({'scope': label, 'direction': direction,
                                                     'slot': slot, 'tensor_key': key})
                        output.row(context, 'tensor', pass_name=pass_name + ':saved_forward', scope=label,
                                   operation='module', direction=direction, slot=slot,
                                   tensor_key=key, **self.numeric_stats(key))
        queue, seen = deque([(scaled_loss.grad_fn, 'loss / actual_group_size')]), set()
        while queue:
            node, inherited_scope = queue.popleft()
            if node is None or node in seen:
                continue
            seen.add(node)
            labels = forward.roots.get(node, [])
            scope = max(labels, key=lambda s: (s.count('.'), len(s))) if labels else inherited_scope
            entry = {'node_id': len(self.nodes), 'operation': node.name(), 'scope': scope,
                     'module_labels': labels, 'saved': {}, 'grad_inputs': [], 'grad_outputs': [],
                     'edges': [], 'pass_name': pass_name}
            if hasattr(node, 'variable'):
                entry['parameter_name'] = getattr(forward, 'parameter_names', {}).get(id(node.variable))
            self.nodes.append(entry)
            for attr in dir(node):
                if attr.startswith('_saved_'):
                    try:
                        entry['saved'][attr] = self.encode(getattr(node, attr))
                    except Exception as error:
                        entry['saved'][attr] = {'capture_error': str(error)}
            for next_node, input_nr in node.next_functions:
                if next_node is not None:
                    entry['edges'].append({'node_name': next_node.name(), 'input_nr': input_nr,
                                           '_node_object': next_node})
                    queue.append((next_node, scope))
            entry['_node_object'] = node
            self.handles.append(node.register_prehook(self.pre(entry)))
            self.handles.append(node.register_hook(self.post(entry)))
        identities = {e['_node_object']: e['node_id'] for e in self.nodes}
        for entry in self.nodes:
            for edge in entry['edges']:
                edge['node_id'] = identities.get(edge.pop('_node_object'))

    def numeric_stats(self, key):
        s = self.archive.statistics[key]
        return {k: s[k] for k in ['max_abs', 'l2', 'nan', 'posinf', 'neginf']}

    def encode(self, value):
        if torch.is_tensor(value):
            return {'tensor_key': self.archive.add(value)}
        if isinstance(value, (list, tuple)):
            return [self.encode(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {'type': type(value).__name__, 'repr': repr(value)}

    def record(self, entry, values, direction):
        keys = []
        for slot, value in enumerate(values):
            if value is None:
                keys.append(None)
                continue
            key = self.archive.add(value)
            keys.append(key)
            self.output.row(self.context, 'tensor', pass_name=self.pass_name,
                            scope=entry['scope'], node_id=entry['node_id'], operation=entry['operation'],
                            direction=direction, slot=slot, tensor_key=key,
                            execution_index=self.execution, **self.numeric_stats(key))
        return keys

    def pre(self, entry):
        def observe(grad_outputs):
            self.execution += 1
            entry['execution_index'] = self.execution
            entry['grad_outputs'] = self.record(entry, grad_outputs, 'entering_backward')
            return None
        return observe

    def post(self, entry):
        def observe(grad_inputs, grad_outputs):
            entry['grad_inputs'] = self.record(entry, grad_inputs, 'returned_by_backward')
            return None
        return observe

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def payload(self):
        return [{k: v for k, v in e.items() if k != '_node_object'} for e in self.nodes]


def compare_tensor(actual, expected):
    if actual is None or expected is None:
        return {'available': False}
    if actual.shape != expected.shape:
        return {'available': False, 'actual_shape': list(actual.shape),
                'expected_shape': list(expected.shape)}
    actual, expected = actual.detach(), expected.detach()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {'available': True, 'all_finite': False,
                'actual': tensor_statistics(actual), 'expected': tensor_statistics(expected)}
    diff = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
    ref = float(expected.abs().max().item()) if expected.numel() else 0.0
    return {'available': True, 'all_finite': True, 'bitwise_equal': torch.equal(actual, expected),
            'max_abs_error': diff, 'relative_to_reference_max': diff / max(ref, 1e-30),
            'actual_max_abs': float(actual.abs().max().item()) if actual.numel() else 0.0,
            'reference_max_abs': ref}


def posthoc_fp32_checks(nodes, archive, device):
    """Only detached FP32 algebra; never reroute model attention or update weights.

    Recomputed scores are labelled reconstructions, not asserted to be the
    hidden fused kernel's exact floating-point intermediates. Dropout-independent
    gradient upper bounds require no guessed dropout mask.
    """
    checks = []
    def value(encoded):
        if isinstance(encoded, dict) and 'tensor_key' in encoded:
            return archive.tensors[encoded['tensor_key']].to(device)
        return encoded
    def grad(keys, index):
        return archive.tensors[keys[index]].to(device) if index < len(keys) and keys[index] else None
    with torch.no_grad():
        for entry in nodes:
            name, saved = entry['operation'], entry['saved']
            item = {'node_id': entry['node_id'], 'scope': entry['scope'], 'operation': name,
                    'analysis_kind': 'detached FP32 algebra after original backward'}
            try:
                incoming = grad(entry['grad_outputs'], 0)
                if incoming is None:
                    continue
                if name.startswith(('MmBackward', 'BmmBackward', 'AddmmBackward')):
                    addmm = name.startswith('AddmmBackward')
                    a = value(saved.get('_saved_mat1' if addmm else '_saved_self'))
                    b = value(saved.get('_saved_mat2'))
                    if not torch.is_tensor(a) and not torch.is_tensor(b):
                        continue
                    alpha = saved.get('_saved_alpha', 1)
                    item['formula'] = 'dA = dY @ B.T * alpha; dB = A.T @ dY * alpha'
                    if torch.is_tensor(b):
                        ga = incoming.matmul(b.transpose(-2, -1)) * alpha
                        item['dA'] = compare_tensor(grad(entry['grad_inputs'], 1 if addmm else 0), ga)
                        item['reference_dA'] = archive.add(ga)
                    if torch.is_tensor(a):
                        gb = a.transpose(-2, -1).matmul(incoming) * alpha
                        item['dB'] = compare_tensor(grad(entry['grad_inputs'], 2 if addmm else 1), gb)
                        item['reference_dB'] = archive.add(gb)
                elif name.startswith('SoftmaxBackward'):
                    p = value(saved.get('_saved_result'))
                    if not torch.is_tensor(p):
                        continue
                    dim = int(saved.get('_saved_dim', -1))
                    if dim > 2**31:
                        dim -= 2**64
                    expected = p * (incoming - (incoming * p).sum(dim=dim, keepdim=True))
                    item['formula'] = 'dS = P * (dP - sum(dP * P, row))'
                    item['dScores'] = compare_tensor(grad(entry['grad_inputs'], 0), expected)
                    item['reference_dScores'] = archive.add(expected)
                elif 'ScaledDotProduct' in name and 'Backward' in name:
                    q = value(saved.get('_saved_query'))
                    k = value(saved.get('_saved_key'))
                    v = value(saved.get('_saved_value'))
                    if not all(torch.is_tensor(x) for x in [q, k, v]):
                        item['limitation'] = 'This version did not expose Q/K/V saved fields.'
                        checks.append(item)
                        continue
                    pdrop = float(saved.get('_saved_dropout_p', 0.0))
                    scale = saved.get('_saved_scale')
                    scale = float(scale) if scale is not None else q.shape[-1] ** -0.5
                    if pdrop >= 1 or q.ndim != 4 or k.shape[-3] != q.shape[-3]:
                        item['limitation'] = 'Upper bounds support this trainer\'s standard non-GQA attention.'
                        checks.append(item)
                        continue
                    returned = [grad(entry['grad_inputs'], j) for j in range(3)]
                    native_output = value(saved.get('_saved_output', saved.get('_saved_out')))
                    heads = []
                    for h in range(q.shape[-3]):
                        qh, kh, vh, gh = [x.select(-3, h) for x in [q, k, v, incoming]]
                        maximum = lambda x: float(x.abs().max().item()) if x.numel() else 0.0
                        mg, mq, mk, mv = [maximum(x) for x in [gh, qh, kh, vh]]
                        lq, lk, dv = q.shape[-2], k.shape[-2], v.shape[-1]
                        bounds = {'Q': 2 * lk * dv * mg * mv * mk * abs(scale) / (1 - pdrop),
                                  'K': 2 * lq * dv * mg * mv * mq * abs(scale) / (1 - pdrop),
                                  'V': lq * mg / (1 - pdrop)}
                        obs = {label: maximum(g.select(-3, h)) if g is not None else None
                               for label, g in zip(['Q', 'K', 'V'], returned)}
                        ratio = {label: obs[label] / max(bound, 1e-30) if obs[label] is not None else None
                                 for label, bound in bounds.items()}
                        heads.append({'head_zero_based': h, 'head_human': h + 1,
                                      'incoming_gradient_max': mg, 'Q_max': mq, 'K_max': mk, 'V_max': mv,
                                      'exact_arithmetic_upper_bounds': bounds, 'observed_grad_max': obs,
                                      'observed_over_bound': ratio,
                                      'large_bound_violation': any(r is not None and r > 1.01 for r in ratio.values())})
                        if torch.is_tensor(native_output) and native_output.shape[-3] == q.shape[-3]:
                            context_max = maximum(native_output.select(-3, h))
                            forward_bound = mv / (1 - pdrop)
                            heads[-1].update(native_attention_output_max=context_max,
                                exact_arithmetic_forward_upper_bound=forward_bound,
                                forward_bound_violation=context_max > forward_bound * 1.01 + 1e-30)
                    item['per_head'] = heads
                    item['bound_note'] = ('Bounds use normalized softmax and dropout scaling. They are '
                        'independent of the dropout mask. Large violations indicate numerical '
                        'inconsistency, not a proof of a particular hidden kernel instruction.')
                    scores = q.matmul(k.transpose(-2, -1)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    item['reconstructed_scores_fp32'] = archive.add(scores)
                    item['stable_softmax_reconstruction_fp32'] = archive.add(probs)
                    item['reconstruction_note'] = ('Posthoc FP32 scores may round differently from '
                        'fused forward/backward GEMM. This is not a dropout-matched backend comparison.')
                    for attr, encoded in saved.items():
                        if 'log' in attr.lower() and 'sum' in attr.lower():
                            lse = value(encoded)
                            if torch.is_tensor(lse) and lse.ndim == scores.ndim - 1:
                                cropped = lse[..., :q.shape[-2]]
                                if cropped.shape == scores.shape[:-1]:
                                    shifted = scores - cropped.unsqueeze(-1)
                                    item['score_minus_saved_lse_fp32'] = archive.add(shifted)
                                    item['exp_score_minus_saved_lse_fp32'] = archive.add(torch.exp(shifted))
                                    item['saved_lse_field'] = attr
                else:
                    continue
                checks.append(item)
            except Exception as error:
                item['analysis_error'] = type(error).__name__ + ': ' + str(error)
                checks.append(item)
    return checks


def sample_record(profile, masked, vocabulary, ids, targets, predictions, selected, loss, divisor, mask_fraction):
    expected_k = max(1, math.floor(len(profile.species_ids) * mask_fraction))
    original = {int(i): value for i, value in zip(profile.species_ids, profile.abundances)}
    selected_again = predictions.detach()[0].index_select(0, ids - 1)
    recomputed_loss = (selected.detach() - targets).square().mean()
    return {'run_id': masked.run_id, 'reported_species_ids': profile.species_ids.tolist(),
            'reported_abundance_text': list(profile.abundance_text),
            'binary_mask': masked.binary_mask.tolist(),
            'visible_species_ids': masked.visible_species_ids.tolist(),
            'visible_abundances': masked.visible_abundances.tolist(),
            'masked_species_ids': masked.masked_species_ids.tolist(),
            'masked_output_indices': (ids - 1).detach().cpu().tolist(),
            'masked_labels': [vocabulary.labels[int(i) - 1] for i in masked.masked_species_ids],
            'target_text': list(masked.target_text), 'targets': targets.detach().cpu().tolist(),
            'selected_predictions': selected.detach().cpu().tolist(),
            'loss': float(loss.detach()), 'actual_group_divisor': divisor,
            'number_masked': int(targets.numel()), 'expected_k': expected_k, 'mask_fraction': mask_fraction,
            'mask_count_correct': int(targets.numel()) == expected_k,
            'mask_removal_correct': np.array_equal(profile.species_ids[masked.binary_mask], masked.visible_species_ids),
            'targets_match_original_values': all(float(original[int(i)]) == float(v)
                                               for i, v in zip(masked.masked_species_ids, masked.targets)),
            'masked_and_visible_ids_disjoint': not bool(set(masked.masked_species_ids) & set(masked.visible_species_ids)),
            'selected_predictions_bitwise_match_output_indices': torch.equal(selected.detach(), selected_again),
            'loss_recomputation_bitwise_equal': torch.equal(loss.detach(), recomputed_loss),
            'max_visible_abundance': float(masked.visible_abundances.max()) if len(masked.visible_abundances) else None,
            'prediction_shape': list(predictions.shape), 'target_shape': list(targets.shape),
            'output_mapping_correct': ids.detach().cpu().tolist() == masked.masked_species_ids.tolist(),
            'loss_definition': 'mean masked squared error / actual accumulation group size for backward'}


def analyse_sample(model, optimizer, generator, forward, profile, masked, vocabulary,
                   values, ids, targets, predictions, selected, loss, scaled_loss, divisor,
                   before_gradients, rng_before, context, output, bundle, alarm,
                   primary_observation=None):
    """Inspect the same retained forward graph, preserving original accumulation."""
    original_accumulation = {n: p.grad for n, p in model.named_parameters()}
    rng_after = capture_rng(generator)
    case = {'context': dict(context), 'sample': sample_record(profile, masked, vocabulary, ids,
                targets, predictions, selected, loss, divisor, bundle['report']['config']['mask_fraction']),
            'snapshot_phase': 'after_original_backward_before_adam',
            'weights_note': 'This model has no running-statistic updates; weights and Adam '
                            'state are unchanged since immediately before this sample.',
            'model_state_before_sample': cpu_tree(model.state_dict()),
            'optimizer_state_before_sample': cpu_tree(optimizer.state_dict()),
            'rng_before_sample_forward': cpu_tree(rng_before),
            'rng_after_original_backward': cpu_tree(rng_after),
            'accumulated_gradients_before_sample': cpu_tree(before_gradients),
            'original_accumulated_gradients_after_sample': cpu_tree(original_accumulation),
            'status': 'pre_analysis_state_saved'}
    bundle['cases'].append(case)
    output.save(bundle)  # Preserve pre-Adam evidence even if detailed tracing fails.
    archive = primary_observation[1] if primary_observation is not None else TensorArchive()
    graph = None
    primary_nodes = primary_observation[0].payload() if primary_observation is not None else None
    if primary_nodes is not None:
        case['original_backward_nodes'] = primary_nodes
        case['original_forward_tensors'] = primary_observation[0].forward_records
        case['individual_gradients_original_backward'] = {
            node['parameter_name']: archive.tensors[node['grad_outputs'][0]]
            for node in primary_nodes if node.get('parameter_name') and node['grad_outputs']
            and node['grad_outputs'][0]}
        case['archive'] = archive.payload()
        output.save(bundle)
    try:
        graph = GraphRecorder(scaled_loss, forward, archive, output, context)
        model.zero_grad(set_to_none=True)
        # Same native graph, saved Q/K/V and saved dropout state. No new forward.
        scaled_loss.backward(retain_graph=True)
        isolated = {n: p.grad for n, p in model.named_parameters()}
        case['isolated_gradients_same_forward_graph'] = cpu_tree(isolated)
        attention = model.blocks[0].self_attn
        width, heads = attention.embed_dim, attention.num_heads
        first_weight_gradient = isolated.get('blocks.0.self_attn.in_proj_weight')
        if first_weight_gradient is not None:
            head_width = width // heads
            case['first_attention_parameter_gradient_heads'] = {
                label: [float(first_weight_gradient[offset + h * head_width:
                    offset + (h + 1) * head_width].detach().abs().max().item())
                    for h in range(heads)]
                for label, offset in [('Q', 0), ('K', width), ('V', 2 * width)]}
            case['first_attention_parameter_gradient_heads_source'] = 'isolated retained-graph repeat'
        checks, all_equal, max_error = [], True, 0.0
        individual_checks = []
        for name, original in original_accumulation.items():
            individual, previous = isolated[name], before_gradients.get(name)
            if original is None and individual is None:
                continue
            if individual is None:
                reconstructed = previous
            else:
                reconstructed = individual if previous is None else previous + individual
            comparison = compare_tensor(original, reconstructed)
            actual_individual = case.get('individual_gradients_original_backward', {}).get(name)
            if actual_individual is not None and individual is not None:
                individual_checks.append({'parameter': name, **compare_tensor(
                    actual_individual, individual.detach().cpu())})
            checks.append({'parameter': name, **comparison})
            all_equal &= bool(comparison.get('bitwise_equal', False))
            max_error = max(max_error, comparison.get('max_abs_error', 0.0))
        case['same_graph_reproduction'] = {'accumulation_bitwise_equal_for_every_parameter': all_equal,
                                          'max_abs_accumulation_error': max_error,
                                          'per_parameter': checks,
                                          'individual_gradient_comparisons': individual_checks,
                                          'note': 'Comparison is previous accumulation + isolated '
                                                  'repeat versus actual first-pass accumulation.'}
        if individual_checks:
            case['same_graph_reproduction']['original_individual_gradients_bitwise_equal'] = all(
                x.get('bitwise_equal', False) for x in individual_checks)
        original_first_weight = case.get('individual_gradients_original_backward', {}).get(
            'blocks.0.self_attn.in_proj_weight')
        if original_first_weight is not None:
            case['first_attention_parameter_gradient_heads'] = {
                label: [float(original_first_weight[offset + h * head_width:
                    offset + (h + 1) * head_width].abs().max().item()) for h in range(heads)]
                for label, offset in [('Q', 0), ('K', width), ('V', 2 * width)]}
            case['first_attention_parameter_gradient_heads_source'] = 'individual gradient from original observed backward'
        case['torch_rng_unchanged_by_repeat'] = rng_equal(rng_after, capture_rng(generator))
        repeat_nodes = graph.payload()
        case['retained_graph_repeat_nodes'] = repeat_nodes
        nodes = primary_nodes if primary_nodes is not None else repeat_nodes
        case['nodes'] = nodes
        case['nodes_pass_name'] = 'original_backward_observed' if primary_nodes is not None else 'retained_graph_backward'
        case['forward_tensors'] = graph.forward_records
        case['fp32_algebra_checks'] = posthoc_fp32_checks(nodes, archive, values.device)
        candidates = []
        for entry in nodes:
            ins = [archive.statistics[k]['max_abs'] for k in entry['grad_outputs'] if k]
            outs = [archive.statistics[k]['max_abs'] for k in entry['grad_inputs'] if k]
            mi, mo = max([v for v in ins if v is not None], default=0.0), max([v for v in outs if v is not None], default=0.0)
            if mo >= alarm and mi < alarm:
                candidates.append({'node_id': entry['node_id'], 'operation': entry['operation'],
                                   'scope': entry['scope'], 'execution_index': entry.get('execution_index'),
                                   'incoming_gradient_max': mi, 'returned_gradient_max': mo})
        case['observed_gradient_growth_boundaries'] = sorted(candidates,
                                  key=lambda e: e.get('execution_index') or 0)
        case['boundary_note'] = 'Growth boundaries are measurements, not automatic proof of numerical error.'
        case['archive'] = archive.payload()
        case['native_saved_scalar_dtype_note'] = (
            'PyTorch may save a Python scaling constant as a zero-dimensional double tensor. '
            'Such native saved constants are preserved verbatim; the diagnostic does not '
            'promote model weights, activations, loss, gradients, or Adam moments.')
        case['status'] = 'analysis_complete'
        print('Detailed sample:', masked.run_id, 'same-graph accumulation match:', all_equal, flush=True)
        for candidate in case['observed_gradient_growth_boundaries'][:8]:
            print('  OBSERVED GROWTH:', json.dumps(candidate), flush=True)
        for check in case['fp32_algebra_checks']:
            for head in check.get('per_head', []):
                if head['large_bound_violation']:
                    print('  ATTENTION GRADIENT BOUND VIOLATION:', check['scope'],
                          'head', head['head_human'], json.dumps(head['observed_over_bound']), flush=True)
    except BaseException as error:
        case['status'] = 'analysis_error'
        case['analysis_error'] = type(error).__name__ + ': ' + str(error)
        case['analysis_traceback'] = traceback.format_exc()
        case['archive'] = archive.payload()
        if graph is not None:
            case['nodes'] = graph.payload()
            case['forward_tensors'] = graph.forward_records
        raise
    finally:
        if graph is not None:
            graph.close()
        for name, p in model.named_parameters():
            p.grad = original_accumulation[name]
        restore_rng(rng_after, generator)
        output.trace.flush()
        output.save(bundle)
    return case


class ReferenceTrace:
    def __init__(self, path):
        self.handle = Path(path).open(encoding='utf-8', newline='') if path else None
        self.reader = csv.DictReader(self.handle, delimiter='\t') if self.handle else None
        self.summary = {'path': str(path) if path else None, 'rows_compared': 0,
                        'identity_mismatches': 0, 'loss_mismatches': 0, 'gradient_mismatches': 0,
                        'first_mismatch': None, 'reference_exhausted': False}

    def compare(self, context, loss, maximum, masked_count, visible_count):
        if self.reader is None:
            return
        ref = next(self.reader, None)
        if ref is None:
            self.summary['reference_exhausted'] = True
            self.reader = None
            return
        self.summary['rows_compared'] += 1
        same_id = (int(ref['epoch']) == context['epoch'] and int(ref['update']) == context['update']
                   and int(ref['sample_position']) == context['sample_position']
                   and ref['run_id'] == context['run_id']
                   and int(ref['masked_count']) == masked_count and int(ref['visible_count']) == visible_count)
        same_loss = math.isclose(float(ref['loss']), loss, rel_tol=2e-6, abs_tol=1e-7)
        same_gradient = math.isclose(float(ref['max_abs_accumulated_gradient']), maximum,
                                     rel_tol=2e-6, abs_tol=1e-7)
        for key, same in [('identity_mismatches', same_id), ('loss_mismatches', same_loss),
                          ('gradient_mismatches', same_gradient)]:
            self.summary[key] += int(not same)
        if not (same_id and same_loss and same_gradient) and self.summary['first_mismatch'] is None:
            self.summary['first_mismatch'] = {'actual_context': dict(context), 'reference_row': dict(ref),
                                               'actual_loss': loss, 'actual_gradient_max': maximum}

    def close(self):
        if self.handle:
            self.handle.close()


def load_configuration(checkpoint_path, input_path, device_name):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Use the original format_version=1 epoch checkpoint.')
    values = dict(checkpoint['config'])
    allowed = {f.name for f in fields(training.Config)}
    if set(values) - allowed:
        raise ValueError('This checkpoint is not from the original trainer configuration.')
    for name in ('head_hidden', 'excluded_features'):
        values[name] = tuple(values[name])
    config = training.Config(**values)
    config = replace(config, input=input_path or config.input, device=device_name)
    config.validate()
    meta = {'vocabulary_labels': tuple(checkpoint['vocabulary_labels']),
            'data_sha256': checkpoint['data_sha256'], 'torch_version': checkpoint.get('torch_version'),
            'config': asdict(config)}
    del checkpoint  # Deliberately do not resume damaged weights or optimizer.
    return config, meta


def environment():
    return {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'default_dtype': str(torch.get_default_dtype()),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'memory_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled()}


@torch.no_grad()
def adam_square_observations(model, beta2):
    """Observe detached FP32 squaring, without altering the actual Adam step."""
    result = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        square = p.grad.detach().square()
        weighted = square * (1 - beta2)
        result.append({'parameter': name, 'dtype': str(square.dtype),
                       'g_squared_inf_count': int(torch.isinf(square).sum().item()),
                       'g_squared_nan_count': int(torch.isnan(square).sum().item()),
                       'weighted_square_inf_count': int(torch.isinf(weighted).sum().item()),
                       'note': 'Detached FP32 g.square(), then weighting. Actual Adam internals '
                               'are assessed separately from its saved post-step state.'})
    return result


def finish_report(bundle):
    report = bundle['report']
    print('\nFINAL FINDINGS', flush=True)
    print('Status:', report['status'], flush=True)
    print('Original trainer SHA-256:', ORIGINAL_TRAINER_SHA256, flush=True)
    print('All model and gradient floating tensors remain FP32. No clipping or backend replacement.', flush=True)
    print('Recording changes graph lifetime (retain_graph=True), not the model equations.', flush=True)
    print('Reference trace comparison:', json.dumps(report.get('reference_trace', {})), flush=True)
    for case in bundle['cases']:
        print('\nSample:', case['sample']['run_id'], 'context:', json.dumps(case['context']), flush=True)
        print('Actual mask and targets:', json.dumps(case['sample'], ensure_ascii=False), flush=True)
        print('Same retained-forward-graph backward reproduced accumulation:',
              case.get('same_graph_reproduction', {}).get('accumulation_bitwise_equal_for_every_parameter'), flush=True)
        print('First encoder Q/K/V parameter gradients by head:',
              json.dumps(case.get('first_attention_parameter_gradient_heads', {})), flush=True)
        print('Head gradient measurement source:',
              case.get('first_attention_parameter_gradient_heads_source'), flush=True)
        boundaries = case.get('observed_gradient_growth_boundaries', [])
        if boundaries:
            print('Observed growth boundaries:', json.dumps(boundaries), flush=True)
        else:
            print('No recorded backward node crossed the configured observation threshold.', flush=True)
        fused = [b for b in boundaries if 'ScaledDotProduct' in b['operation']]
        if fused:
            print('The measured boundary is fused attention backward. Its hidden internal '
                  'multiplications are not individually exposed by autograd. Reconstructed '
                  'scores and mathematical bounds are supporting diagnostics, not measurements '
                  'of a hidden GPU instruction.', flush=True)
        violations = [(c['node_id'], h['head_human'], h['observed_over_bound'])
                      for c in case.get('fp32_algebra_checks', []) for h in c.get('per_head', [])
                      if h['large_bound_violation']]
        print('Attention gradient upper-bound violations:', json.dumps(violations), flush=True)
        print('DETACHED FP32 OPERATOR CHECK DETAILS (not substituted into training):', flush=True)
        for check in case.get('fp32_algebra_checks', []):
            print(json.dumps(check, ensure_ascii=False), flush=True)
        print('FP32 gradient squaring observations:', json.dumps([
            x for x in case.get('fp32_adam_square_observations', [])
            if x['g_squared_inf_count'] or x['g_squared_nan_count']]), flush=True)
    print('\nAdam result:', json.dumps(report.get('adam_result', {})), flush=True)
    print('A large gradient alone does not establish an incorrect derivative. '
          'A repeated GPU event is not guaranteed by a matching seed. '
          'This run must be assessed using its recorded trace and reproduction checks.', flush=True)


def replay(args, output):
    config, checkpoint_meta = load_configuration(args.checkpoint, args.input, args.device)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('The default floating dtype must be float32 for this replay.')
    vocabulary = training.build_vocabulary(config.input, config.excluded_features)
    if vocabulary.labels != checkpoint_meta['vocabulary_labels']:
        raise ValueError('Species vocabulary or order differs from the original checkpoint.')
    profiles, audit = training.load_profiles(config.input, vocabulary)
    if audit['data_sha256'] != checkpoint_meta['data_sha256']:
        raise ValueError('Retained original profile values/order differ from the reference checkpoint.')
    if audit['empty_runs']:
        raise ValueError('Input includes empty runs; no sample was silently discarded.')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested CPU/CUDA device is unavailable.')
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = np.random.default_rng(config.seed)
    model = training.build_model(config, vocabulary.size, device)
    optimizer = training.build_optimizer(model, config)
    training.make_diagnostic_plan(profiles, config)
    model.train()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Model is not entirely float32.')
    reference_path = args.reference_trace
    if reference_path is None:
        candidate = Path(args.checkpoint).resolve().parent.parent.parent / 'nonfinite_replay' / 'diagnostic_trace.tsv'
        if candidate.exists():
            reference_path = str(candidate)
    reference = ReferenceTrace(reference_path)
    report = {'status': 'running', 'config': asdict(config), 'input_audit': audit,
              'reference_checkpoint': str(Path(args.checkpoint).resolve()),
              'reference_checkpoint_torch': checkpoint_meta['torch_version'], 'environment': environment(),
              'original_trainer_sha256': ORIGINAL_TRAINER_SHA256,
              'original_checkpoint_weights_loaded': False, 'original_checkpoint_optimizer_loaded': False,
              'training_dtype': 'float32', 'gradient_clipping': False,
              'attention_backend': 'unchanged; original automatic backend selection',
              'analysis_precision': 'FP32 tensor arithmetic; rescaled FP32 diagnostic reductions',
              'graph_lifetime_change': 'retain_graph=True to inspect the actual saved forward calculation',
              'alarm_max_abs': args.alarm_max_abs,
              'alarm_note': 'Observation trigger only. It never clips, rescales, or replaces gradients.',
              'target': {'epoch': args.target_epoch, 'update': args.target_update, 'run_id': args.target_run},
              'max_updates': args.max_updates, 'reference_trace': reference.summary}
    bundle = {'diagnostic_format': 2, 'report': report, 'cases': [],
              'vocabulary_labels': vocabulary.labels, 'original_trainer_source': ORIGINAL_TRAINER_SOURCE,
              'script_source': Path(__file__).read_text(encoding='utf-8'), 'replay_rows': []}
    output.save(bundle)
    print('FLOAT32 ORIGINAL-REPLAY OPERATION TRACE', flush=True)
    print('Original trainer SHA-256 verified:', ORIGINAL_TRAINER_SHA256, flush=True)
    print('Checkpoint is configuration/data reference only; initialization starts at epoch 1.', flush=True)
    print('Device:', device, 'runs:', len(profiles), 'vocabulary:', vocabulary.size, flush=True)
    print('Float32 model, loss, gradients and Adam; no clipping; original attention selection.', flush=True)
    print('Retained native forward graphs keep the actual dropout choices for detailed tracing.', flush=True)
    print('Output:', output.folder, flush=True)
    forward = ForwardRecorder(model)
    started, update, total_seen, context = time.perf_counter(), 0, 0, {}
    group_records = []
    try:
        for epoch in range(1, args.epochs + 1):
            order = generator.permutation(len(profiles))
            seen = 0
            for start in range(0, len(order), config.accumulate_samples):
                group = order[start:start + config.accumulate_samples]
                optimizer.zero_grad(set_to_none=True)
                group_records, captured_this_group, group_cases = [], False, []
                for group_position, index in enumerate(group, 1):
                    profile = profiles[int(index)]
                    masked = training.mask_profile(profile, config.mask_fraction, generator)
                    context = {'epoch': epoch, 'update': update + 1, 'sample_position': seen + 1,
                               'run_id': masked.run_id, 'group_position': group_position,
                               'group_size': len(group)}
                    rng_before = capture_rng(generator)
                    # Keep exact previous accumulation; weights/Adam remain fixed within this group.
                    before = {n: p.grad.detach().clone() if p.grad is not None else None
                              for n, p in model.named_parameters()}
                    forward.reset()
                    values, ids, targets = training.prepare_batch(masked, device)
                    predictions = model(values)
                    loss, selected = training.compute_masked_loss(predictions, ids, targets)
                    loss_value = float(loss.detach())
                    scaled_loss = loss / len(group)
                    if not math.isfinite(loss_value):
                        # Preserve the original trainer's stop BEFORE backward on invalid loss.
                        archive = TensorArchive()
                        graph = GraphRecorder(scaled_loss, forward, archive, output, context)
                        try:
                            case = {'context': dict(context),
                                'sample': sample_record(profile, masked, vocabulary, ids, targets,
                                    predictions, selected, loss, len(group), config.mask_fraction),
                                'status': 'nonfinite_forward_loss',
                                'snapshot_phase': 'after_original_forward_before_backward_and_adam',
                                'model_state_before_sample': cpu_tree(model.state_dict()),
                                'optimizer_state_before_sample': cpu_tree(optimizer.state_dict()),
                                'accumulated_gradients_before_sample': cpu_tree(before),
                                'rng_before_sample_forward': cpu_tree(rng_before),
                                'nodes': graph.payload(), 'forward_tensors': graph.forward_records,
                                'archive': archive.payload(),
                                'note': 'No backward was run on the invalid loss, matching the original trainer.'}
                            bundle['cases'].append(case)
                        finally:
                            graph.close()
                        report.update(status='nonfinite_forward_loss', failure_context=dict(context))
                        return bundle
                    watched = (epoch == args.target_epoch and update + 1 == args.target_update
                               and (args.target_run == '*' or masked.run_id == args.target_run))
                    primary_observation = None
                    if watched:
                        primary_archive = TensorArchive()
                        primary_graph = GraphRecorder(scaled_loss, forward, primary_archive, output,
                                                      context, pass_name='original_backward_observed')
                        primary_observation = (primary_graph, primary_archive)
                    try:
                        scaled_loss.backward(retain_graph=True)
                    finally:
                        if primary_observation is not None:
                            primary_graph.close()
                    gradients = scan((n, p.grad) for n, p in model.named_parameters())
                    seen += 1
                    total_seen += 1
                    row = {**context, 'masked_count': len(ids), 'visible_count': values.shape[1],
                           'loss': loss_value, 'max_abs_accumulated_gradient': gradients['max_abs'],
                           'gradient_nonfinite': gradients['nonfinite']}
                    bundle['replay_rows'].append(row)
                    output.row(context, 'sample', pass_name='original_replay',
                               max_abs=gradients['max_abs'],
                               masked_count=len(ids), visible_count=values.shape[1], loss=loss_value,
                               gradient_nonfinite=gradients['nonfinite'])
                    reference.compare(context, loss_value, gradients['max_abs'], len(ids), values.shape[1])
                    group_records.append({'context': dict(context), 'run_id': masked.run_id,
                        'visible_species_ids': masked.visible_species_ids.tolist(),
                        'visible_abundances': masked.visible_abundances.tolist(),
                        'masked_species_ids': masked.masked_species_ids.tolist(),
                        'target_text': list(masked.target_text), 'targets': masked.targets.tolist(),
                        'loss': loss_value, 'gradient_max': gradients['max_abs']})
                    spike = bool(gradients['nonfinite']) or not math.isfinite(loss_value) or gradients['max_abs'] >= args.alarm_max_abs
                    if (watched or spike) and (not captured_this_group or spike and not any(c.get('original_spike') for c in group_cases)):
                        print('CAPTURE:', context, 'loss=', loss_value,
                              'max|accumulated gradient|=', gradients['max_abs'], flush=True)
                        case = analyse_sample(model, optimizer, generator, forward, profile, masked,
                            vocabulary, values, ids, targets, predictions, selected, loss, scaled_loss,
                            len(group), before, rng_before, context, output, bundle, args.alarm_max_abs,
                            primary_observation=primary_observation)
                        case['original_spike'] = spike
                        case['accumulation_group_before_this_sample'] = copy.deepcopy(group_records[:-1])
                        group_cases.append(case)
                        captured_this_group = True
                    del before, scaled_loss, loss, selected, predictions, primary_observation
                    if watched:
                        del primary_archive, primary_graph
                    forward.reset()
                    if gradients['nonfinite'] or not math.isfinite(loss_value):
                        report.update(status='nonfinite_before_adam', failure_context=dict(context))
                        return bundle
                update += 1
                learning_rate = training.transformer_learning_rate(update, config)
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = learning_rate
                for case in group_cases:
                    case['accumulation_group_complete'] = copy.deepcopy(group_records)
                    case['actual_accumulated_gradients_before_adam'] = cpu_tree({n: p.grad for n, p in model.named_parameters()})
                    case['adam_learning_rate'] = learning_rate
                if group_cases:
                    square_observations = adam_square_observations(model, config.adam_beta2)
                    for case in group_cases:
                        case['fp32_adam_square_observations'] = square_observations
                    # The original state before step must survive an optimizer failure.
                    output.save(bundle)
                optimizer.step()  # Same original Adam implementation/settings; still no clipping.
                parameter_stats = scan(model.named_parameters())
                optimizer_stats = scan(optimizer_tensors(model, optimizer))
                if group_cases or parameter_stats['nonfinite'] or optimizer_stats['nonfinite']:
                    result = {'update': update, 'learning_rate': learning_rate,
                              'parameters': parameter_stats, 'optimizer_state': optimizer_stats}
                    report['adam_result'] = result
                    for case in group_cases:
                        case['adam_result'] = result
                    bundle['model_state_after_observed_adam_step'] = cpu_tree(model.state_dict())
                    bundle['optimizer_state_after_observed_adam_step'] = cpu_tree(optimizer.state_dict())
                if parameter_stats['nonfinite'] or optimizer_stats['nonfinite']:
                    report.update(status='nonfinite_after_adam', failure_context=dict(context))
                    return bundle
                if any(case.get('original_spike') for case in group_cases):
                    report['status'] = 'finite_gradient_spike_captured'
                    return bundle
                if seen % args.progress_every < len(group) or seen == len(profiles):
                    print(f'Replay epoch {epoch:03d}: {seen:,}/{len(profiles):,}; update={update:,}; '
                          f'max|accumulated gradient|={gradients["max_abs"]:.6g}; lr={learning_rate:.6g}', flush=True)
                    output.trace.flush()
                if update >= args.max_updates:
                    report['status'] = ('finite_gradient_spike_captured' if any(c.get('original_spike') for c in bundle['cases'])
                                        else 'recorded_failure_not_reproduced_within_update_limit')
                    return bundle
            report['last_completed_epoch'] = epoch
        report['status'] = 'replay_completed_without_recorded_failure'
        return bundle
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'execution_error',
                      failure_context=dict(context), error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc())
        try:
            bundle['state_at_execution_error_or_interrupt'] = {
                'context': dict(context), 'model_state': cpu_tree(model.state_dict()),
                'optimizer_state': cpu_tree(optimizer.state_dict()),
                'gradients_at_error': cpu_tree({n: p.grad for n,p in model.named_parameters()}),
                'rng_at_error': cpu_tree(capture_rng(generator)),
                'group_records_completed': copy.deepcopy(group_records),
                'note': 'Partial state at interruption; not asserted to be pre-sample state.'}
            emergency = bundle['state_at_execution_error_or_interrupt']
            if locals().get('rng_before') is not None:
                emergency['rng_before_current_sample'] = cpu_tree(rng_before)
            if locals().get('before') is not None:
                emergency['accumulation_before_current_sample'] = cpu_tree(before)
            observed = locals().get('primary_observation')
            if observed is not None:
                emergency['original_backward_nodes'] = observed[0].payload()
                emergency['archive'] = observed[1].payload()
        except Exception as snapshot_error:
            report['emergency_snapshot_error'] = type(snapshot_error).__name__ + ': ' + str(snapshot_error)
        raise
    finally:
        forward.close()
        reference.close()
        report.update(seconds=time.perf_counter() - started, samples_processed=total_seen,
                      completed_optimizer_updates=update, reference_trace=reference.summary)
        finish_report(bundle)
        output.save(bundle)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='Trusted ORIGINAL epoch checkpoint; configuration/data identity only, not resume.')
    parser.add_argument('--input', default='non_asd_qc_pass_reported_required.tsv.gz')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default=None, help='New empty diagnostic directory.')
    parser.add_argument('--reference-trace', help='Old diagnostic_trace.tsv; compared row by row if supplied.')
    parser.add_argument('--epochs', type=int, default=5, help='Maximum replay epochs; also stops at max-updates.')
    parser.add_argument('--max-updates', type=int, default=4113)
    parser.add_argument('--target-epoch', type=int, default=2)
    parser.add_argument('--target-update', type=int, default=4113)
    parser.add_argument('--target-run', default='SRR25256440', help='Use * to inspect first sample of target update.')
    parser.add_argument('--alarm-max-abs', type=float, default=1e6,
                        help='Observation trigger, NOT clipping. Default 1e6 catches the recorded 4e20 spike.')
    parser.add_argument('--progress-every', type=int, default=1000)
    args = parser.parse_args()
    if min(args.epochs, args.max_updates, args.target_epoch, args.target_update, args.progress_every) < 1:
        parser.error('Epochs, updates and progress intervals must be positive.')
    if not math.isfinite(args.alarm_max_abs) or args.alarm_max_abs <= 0:
        parser.error('alarm-max-abs must be finite and positive.')
    return args


def main():
    args = parse_arguments()
    folder = args.output_dir or ('attention_diagnostic_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.parent.name == 'checkpoints':
        old_results = checkpoint.parent.parent
        if Path(folder).resolve() == old_results or old_results in Path(folder).resolve().parents:
            raise ValueError('Choose a new diagnostic folder outside the existing training results.')
    output = OutputFiles(folder)
    console_out, console_err = sys.stdout, sys.stderr
    code = 0
    try:
        with redirect_stdout(Tee(console_out, output.report)), redirect_stderr(Tee(console_err, output.report)):
            output.save({'diagnostic_format': 2, 'report': {'status': 'preflight', 'arguments': vars(args)}, 'cases': []})
            try:
                result = replay(args, output)
                code = 2 if result['report']['status'].startswith('nonfinite_') else 0
            except KeyboardInterrupt:
                output.last_payload['report']['status'] = 'interrupted'
                output.save(output.last_payload)
                print('Interrupted. Saved available evidence in diagnostic_bundle.pt.', flush=True)
                code = 130
            except Exception:
                output.last_payload['report'].update(status='execution_error', traceback=traceback.format_exc())
                output.save(output.last_payload)
                traceback.print_exc()
                code = 1
    finally:
        output.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
