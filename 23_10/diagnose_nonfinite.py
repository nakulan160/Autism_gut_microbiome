#!/usr/bin/env python3
r"""Inspect and trace a nonfinite GMrepo training failure without editing training.

Put this file beside train_masked_abundance.py, then run:

  python diagnose_nonfinite.py inspect --checkpoint results/checkpoints/epoch_0005.pt

For the first failing update, replay the original initialization and configuration:

  python -u diagnose_nonfinite.py replay \
    --checkpoint results/checkpoints/epoch_0005.pt \
    --input non_asd_qc_pass_reported_blank.tsv.gz --device cuda

Replay starts at epoch ONE, not at the damaged checkpoint. It uses the checkpoint
only to recover configuration and verify data/vocabulary identity. It preserves
the model, data, masking, optimizer settings, schedule and accumulation rule.
It adds numerical checks without clipping, replacing values or skipping samples.
Reports go to a NEW directory; existing results and checkpoints are read-only.
Replay is slower than ordinary training. GPU execution need not be bitwise
reproducible, even with the same seed. A successful replay cannot prove that the
earlier execution had no numerical fault.

At the first detected failure, the default snapshot can use about 450 MB for the
full-size model. --no-snapshot disables it. NaN anomaly tracing and explicit Inf
checks are then tried on the failing sample, where pre-update weights are intact.
Only load checkpoints/snapshots that you created or otherwise trust.

Exit codes: 0 = inspection/replay finished; 2 = replay detected a numerical failure;
1 = setup or execution error; 130 = interrupted. A report is also written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import traceback
from contextlib import redirect_stderr
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import train_masked_abundance as training


def tensor_summary(value: torch.Tensor) -> dict:
    value = value.detach().cpu()
    finite = torch.isfinite(value)
    kept = value[finite]
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "elements": value.numel(), "nan": int(torch.isnan(value).sum()),
        "positive_inf": int(torch.isposinf(value).sum()),
        "negative_inf": int(torch.isneginf(value).sum()),
        "finite_abs_max": float(kept.abs().max()) if kept.numel() else None,
    }


def tensor_maxima(named_tensors) -> list[tuple[str, torch.Tensor, float]]:
    """Infinity norms avoid overflowing a sum of squares just to check gradients."""
    buckets = {}
    empty = []
    for name, value in named_tensors:
        if value is None:
            continue
        value = value.detach()
        if not value.numel():
            empty.append((name, value, 0.0))
            continue
        buckets.setdefault((value.device, value.dtype), []).append((name, value))
    result = empty
    for entries in buckets.values():
        tensors = [value for _, value in entries]
        if hasattr(torch, "_foreach_norm"):
            norms = torch._foreach_norm(tensors, float("inf"))
        else:
            norms = [value.abs().amax() for value in tensors]
        maxima = torch.stack(tuple(norms)).cpu().tolist()
        result.extend((name, value, float(maximum))
                      for (name, value), maximum in zip(entries, maxima))
    return result


def scan(named_tensors) -> tuple[dict, dict[str, float]]:
    maxima = tensor_maxima(named_tensors)
    bad = [{"name": name, **tensor_summary(value)}
           for name, value, maximum in maxima if not math.isfinite(maximum)]
    finite = sorted(((maximum, name) for name, _, maximum in maxima
                     if math.isfinite(maximum)), reverse=True)
    return ({"nonfinite": bad, "max_finite_abs": finite[0][0] if finite else None,
             "largest_finite_tensors": [{"name": name, "abs_max": maximum}
                                         for maximum, name in finite[:5]]},
            {name: maximum for name, _, maximum in maxima})


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(checkpoint: dict):
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected this training script's format_version=1 checkpoint.")
    values = dict(checkpoint["config"])
    allowed = {field.name for field in fields(training.Config)}
    if set(values) - allowed:
        raise ValueError("The checkpoint configuration differs from this training script.")
    for name in ("excluded_features", "head_hidden"):
        values[name] = tuple(values[name])
    config = training.Config(**values)
    config.validate()
    return config


def new_output(args) -> Path:
    name = args.output_dir or ("nonfinite_" + args.mode + "_" +
                               datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out = Path(name).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.parent.name == "checkpoints":
        original_results = checkpoint.parent.parent
        if out == original_results or original_results in out.parents:
            raise ValueError("Choose a new diagnostic folder outside the existing results folder.")
    out.mkdir(parents=True, exist_ok=False)
    return out


def environment() -> dict:
    return {
        "python": sys.version, "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "trainer_sha256": hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def parameter_mapping(model, optimizer, saved_optimizer) -> dict:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    current_groups, saved_groups = optimizer.param_groups, saved_optimizer["param_groups"]
    if len(current_groups) != len(saved_groups):
        raise ValueError("Optimizer group count differs from the original training script.")
    mapping = {}
    for current, saved in zip(current_groups, saved_groups):
        if len(current["params"]) != len(saved["params"]):
            raise ValueError("Optimizer parameter order cannot be mapped to this model.")
        for parameter, saved_id in zip(current["params"], saved["params"]):
            mapping[saved_id] = names[id(parameter)]
    return mapping


def inspect(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = configuration(checkpoint)
    torch.set_num_threads(config.threads)
    model = training.build_model(config, len(checkpoint["vocabulary_labels"]), torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = training.build_optimizer(model, config)
    mapping = parameter_mapping(model, optimizer, checkpoint["optimizer_state_dict"])
    weights, _ = scan(model.named_parameters())
    state_tensors = []
    for parameter_id, state in checkpoint["optimizer_state_dict"]["state"].items():
        if parameter_id not in mapping:
            raise ValueError(f"Unmapped optimizer parameter ID {parameter_id}.")
        for key, value in state.items():
            if torch.is_tensor(value):
                state_tensors.append((f"{mapping[parameter_id]}/{key}", value))
    states, _ = scan(state_tensors)
    out = new_output(args)
    report = {
        "mode": "inspect", "checkpoint": str(Path(args.checkpoint).resolve()),
        "completed_epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "checkpoint_torch": checkpoint.get("torch_version"), "environment": environment(),
        "model_parameters": weights, "optimizer_state": states,
        "optimizer_parameter_mapping": {str(key): value for key, value in mapping.items()},
        "full_state_finite": not weights["nonfinite"] and not states["nonfinite"],
        "limitation": "A checkpoint shows existing damage, not the earlier operation that caused it.",
    }
    write_report(out / "inspection.json", report)
    print(f"Checkpoint: {args.checkpoint}; completed epoch: {checkpoint['epoch']}", flush=True)
    for item in weights["nonfinite"] + states["nonfinite"]:
        print(f"BAD {item['name']}: NaN={item['nan']}, +Inf={item['positive_inf']}, "
              f"-Inf={item['negative_inf']}; finite abs max={item['finite_abs_max']}", flush=True)
    print(f"Full checkpoint state finite: {report['full_state_finite']}", flush=True)
    print(f"Inspection report: {out / 'inspection.json'}", flush=True)
    return report


def optimizer_tensors(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                yield f"{names[id(parameter)]}/{key}", value


def capture_rng():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class NonfiniteBoundary(RuntimeError):
    pass


def anomaly_probe(model, masked, divisor, device, rng_before, path: Path) -> dict:
    """Replay one forward/backward with NaN tracing and explicit Inf boundary checks."""
    handles = []

    def check(value, label):
        if not bool(torch.isfinite(value).all()):
            raise NonfiniteBoundary(f"Nonfinite tensor at {label}: {tensor_summary(value)}")

    def hook_for(name):
        def forward_hook(module, inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            for index, value in enumerate(values):
                if torch.is_tensor(value):
                    check(value, f"forward output of {name}[{index}]")
                    if value.requires_grad:
                        def gradient_hook(gradient, label=f"output gradient of {name}[{index}]"):
                            check(gradient, label)
                            return gradient
                        value.register_hook(gradient_hook)
        return forward_hook

    model.zero_grad(set_to_none=True)
    restore_rng(rng_before)
    model.train()
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook_for(name)))
    result = {"single_sample_probe": True,
              "limitation": "This isolates one sample, not prior gradient accumulation; "
                            "instrumentation/GPU execution may affect reproducibility."}
    with path.open("w", encoding="utf-8") as log, redirect_stderr(log):
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                values, ids, targets = training.prepare_batch(masked, device)
                loss, _ = training.compute_masked_loss(model(values), ids, targets)
                check(loss, "masked MSE")
                (loss / divisor).backward()
            result["gradient_scan"], _ = scan((name, p.grad) for name, p in model.named_parameters())
            result["result"] = "Single-sample forward/backward completed. Inspect gradient_scan."
        except Exception as error:
            traceback.print_exc(file=log)
            result["result"] = f"{type(error).__name__}: {error}"
        finally:
            for handle in handles:
                handle.remove()
    return result


def replay(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_config = configuration(checkpoint)
    input_path = args.input or original_config.input
    epochs = args.epochs or int(checkpoint["epoch"])
    if epochs < 1 or args.progress_every < 1:
        raise ValueError("Epochs and progress-every must be positive.")
    expected_labels = tuple(checkpoint["vocabulary_labels"])
    expected_data = checkpoint["data_sha256"]
    checkpoint_version = checkpoint.get("torch_version")
    # The saved weights and optimizer state are deliberately NOT loaded for replay.
    del checkpoint
    config = replace(original_config, input=input_path, epochs=epochs)
    vocabulary = training.build_vocabulary(config.input, config.excluded_features)
    if vocabulary.labels != expected_labels:
        raise ValueError("Input vocabulary identities/order differ from the reference checkpoint.")
    profiles, audit = training.load_profiles(config.input, vocabulary)
    if audit["data_sha256"] != expected_data:
        raise ValueError("Input retained profiles differ from the reference checkpoint.")
    if audit["empty_runs"]:
        raise ValueError("An input run has no retained targets.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but not available in this environment.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, or cuda:N.")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)
    model = training.build_model(config, vocabulary.size, device)
    optimizer = training.build_optimizer(model, config)
    training.make_diagnostic_plan(profiles, config)
    out = new_output(args)
    report = {
        "mode": "replay", "status": "running", "environment": environment(),
        "reference_checkpoint": str(Path(args.checkpoint).resolve()),
        "reference_checkpoint_torch": checkpoint_version, "config": asdict(config),
        "replay_starts_at_epoch": 1, "checkpoint_weights_loaded": False,
        "checkpoint_optimizer_loaded": False, "data_sha256": audit["data_sha256"],
        "vocabulary_sha256": vocabulary.sha256,
        "limitation": "Same seed/configuration is not a guarantee of identical GPU execution. "
                      "No detected failure does not certify the earlier run.",
    }
    report_path = out / "replay_report.json"
    write_report(report_path, report)
    print(f"DIAGNOSTIC REPLAY FROM EPOCH 1; seed={config.seed}; device={device}", flush=True)
    print("Reference checkpoint is read-only; damaged optimizer state is not resumed.", flush=True)
    print(f"Output folder: {out}", flush=True)
    update = total_seen = 0
    phase = "initialization"
    masked = rng_before = None
    context = {}
    gradient_stats = {"nonfinite": []}
    gradient_maxima = {}
    started = time.perf_counter()

    def record_failure():
        report["status"] = "nonfinite_detected"
        report["first_detected_phase"] = phase
        report["context"] = context
        report["gradients"] = gradient_stats
        report["seconds"] = time.perf_counter() - started
        if masked is not None:
            report["sample"] = {
                "run_id": masked.run_id, "visible_species_ids": masked.visible_species_ids.tolist(),
                "visible_abundances": masked.visible_abundances.tolist(),
                "masked_species_ids": masked.masked_species_ids.tolist(),
                "target_text": list(masked.target_text), "targets": masked.targets.tolist(),
            }
        report["model_parameters"], _ = scan(model.named_parameters())
        report["optimizer_state"], _ = scan(optimizer_tensors(model, optimizer))
        overflow_candidates = []
        for name, parameter in model.named_parameters():
            maximum = gradient_maxima.get(name)
            if maximum is not None and math.isfinite(maximum):
                weighted_square = (1 - config.adam_beta2) * maximum * maximum
                limit = torch.finfo(parameter.dtype).max
                if weighted_square > limit:
                    overflow_candidates.append({"parameter": name, "gradient_abs_max": maximum,
                                                "weighted_squared_gradient_float64": weighted_square,
                                                "parameter_dtype_max": limit})
        report["squared_gradient_overflow_candidates"] = overflow_candidates
        write_report(report_path, report)
        if not args.no_snapshot:
            torch.save({
                "diagnostic_format": 1, "report": report,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "gradients": {name: parameter.grad for name, parameter in model.named_parameters()
                              if parameter.grad is not None},
                "torch_rng_before_sample": rng_before,
                "snapshot_phase": phase,
            }, out / "first_failure_state.pt")
            report["failure_snapshot"] = "first_failure_state.pt"
        if masked is not None and rng_before is not None and phase != "after_adam_step":
            report["anomaly_probe"] = anomaly_probe(model, masked, len(group), device,
                                                     rng_before, out / "anomaly_trace.txt")
        write_report(report_path, report)
        print(f"FIRST FAILURE: {phase}; {context}", flush=True)
        for category in ("gradients", "optimizer_state", "model_parameters"):
            for item in report.get(category, {}).get("nonfinite", []):
                print(f"  {category}: {item['name']} "
                      f"NaN={item['nan']} +Inf={item['positive_inf']} -Inf={item['negative_inf']}",
                      flush=True)
        if overflow_candidates:
            print("Some finite gradients are large enough to overflow Adam's squared history.", flush=True)
        print(f"Failure report: {report_path}", flush=True)
        if "anomaly_probe" in report:
            print(f"Anomaly trace: {out / 'anomaly_trace.txt'}", flush=True)
        return report

    trace_path = out / "diagnostic_trace.tsv"
    try:
        with trace_path.open("w", newline="", encoding="utf-8") as trace:
            writer = csv.writer(trace, delimiter="\t", lineterminator="\n")
            writer.writerow(("epoch", "update", "sample_position", "run_id", "masked_count",
                             "visible_count", "loss", "max_abs_accumulated_gradient"))
            for epoch in range(1, epochs + 1):
                model.train()
                order = rng.permutation(len(profiles))
                seen = 0
                for start in range(0, len(order), config.accumulate_samples):
                    group = order[start:start + config.accumulate_samples]
                    optimizer.zero_grad(set_to_none=True)
                    group_ids = []
                    for index in group:
                        masked = training.mask_profile(profiles[int(index)], config.mask_fraction, rng)
                        group_ids.append(masked.run_id)
                        context = {"epoch": epoch, "optimizer_update": update + 1,
                                   "sample_position": seen + 1, "run_id": masked.run_id,
                                   "accumulation_group_run_ids_so_far": list(group_ids),
                                   "group_size": len(group)}
                        rng_before = capture_rng()
                        values, ids, targets = training.prepare_batch(masked, device)
                        phase = "forward_or_masked_loss"
                        predictions = model(values)
                        loss, selected = training.compute_masked_loss(predictions, ids, targets)
                        loss_value = float(loss.detach())
                        if not math.isfinite(loss_value):
                            context["loss"] = str(loss_value)
                            report["predictions"] = tensor_summary(predictions)
                            report["targets"] = tensor_summary(targets)
                            return record_failure()
                        phase = "backward_or_gradient_accumulation"
                        (loss / len(group)).backward()
                        gradient_stats, gradient_maxima = scan((name, parameter.grad)
                                                              for name, parameter in model.named_parameters())
                        if gradient_stats["nonfinite"]:
                            context["loss_before_backward"] = loss_value
                            return record_failure()
                        seen += 1
                        total_seen += 1
                        writer.writerow((epoch, update + 1, seen, masked.run_id, len(ids),
                                         values.shape[1], format(loss_value, ".9g"),
                                         gradient_stats["max_finite_abs"]))
                    update += 1
                    learning_rate = training.transformer_learning_rate(update, config)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate
                    context["learning_rate"] = learning_rate
                    phase = "after_adam_step"
                    optimizer.step()
                    parameter_stats, _ = scan(model.named_parameters())
                    optimizer_stats, _ = scan(optimizer_tensors(model, optimizer))
                    if parameter_stats["nonfinite"] or optimizer_stats["nonfinite"]:
                        return record_failure()
                    trace.flush()
                    if seen % args.progress_every < len(group) or seen == len(profiles):
                        print(f"Replay epoch {epoch:03d}: {seen:,}/{len(profiles):,}; "
                              f"update={update:,}; max|accumulated gradient|="
                              f"{gradient_stats['max_finite_abs']:.6g}; lr={learning_rate:.6g}", flush=True)
                report["last_completed_epoch"] = epoch
                report["optimizer_updates"] = update
                write_report(report_path, report)
        report.update(status="no_nonfinite_detected", seconds=time.perf_counter() - started,
                      samples_processed=total_seen)
        write_report(report_path, report)
        print(f"Replay completed without a detected nonfinite tensor. Report: {report_path}", flush=True)
        return report
    except KeyboardInterrupt:
        report.update(status="interrupted", context=context,
                      seconds=time.perf_counter() - started)
        write_report(report_path, report)
        raise
    except Exception as error:
        report.update(status="execution_error", first_detected_phase=phase, context=context,
                      error=f"{type(error).__name__}: {error}")
        (out / "execution_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_report(report_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inspect", "replay"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--output-dir", help="New directory outside the existing training results.")
        if mode == "replay":
            sub.add_argument("--input", help="Original or equivalently filtered reported-abundance TSV(.gz).")
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--epochs", type=int, help="Defaults to the reference checkpoint epoch.")
            sub.add_argument("--progress-every", type=int, default=1000)
            sub.add_argument("--no-snapshot", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect(args) if args.mode == "inspect" else replay(args)
        return 2 if report.get("status") == "nonfinite_detected" else 0
    except KeyboardInterrupt:
        print("Diagnostic interrupted; original training results were not modified.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"DIAGNOSTIC ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
