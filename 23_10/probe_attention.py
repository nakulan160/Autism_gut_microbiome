#!/usr/bin/env python3
r"""Compare attention kernels on the sample that first produced extreme gradients.

Place this beside train_masked_abundance.py. Run in the original GPU environment:

  python -u probe_attention.py \
    --snapshot nonfinite_replay/first_failure_state.pt \
    --input non_asd_qc_pass_reported_required.tsv.gz --device cuda

This is a focused numerical probe, NOT a training resume or repair. No optimizer
is loaded or stepped. All possible one-entry masks of SRR25256440 are tested.
The original data/vocabulary are verified, and reports go to a new directory.

Important limits:
* Snapshot weights are AFTER the first failed Adam step, not the exact weights
  immediately before SRR25256440's original backward pass.
* Evaluation mode disables dropout for this diagnostic comparison only.
* First-layer Q/K/V and incoming output gradients are captured from a MATH FP32
  model pass. Each isolated attention backend then receives the same tensors and
  incoming gradient, with dropout=0. Float64 MATH supplies the reference.
* A discrepancy establishes a numerical problem on these probe inputs. A clean
  probe cannot rule out the original training-mode failure.
* This script never changes the training program, input data or original results.

Use only your own trusted diagnostic snapshots. Requires the same dependencies
as training and PyTorch >=2.2. CPU mode is available for checking the script but
does not test the A100 kernels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import train_masked_abundance as training


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    selected = value[finite]
    return {
        "shape": list(value.shape), "strides": list(value.stride()),
        "dtype": str(value.dtype), "nan": int(torch.isnan(value).sum()),
        "positive_inf": int(torch.isposinf(value).sum()),
        "negative_inf": int(torch.isneginf(value).sum()),
        "finite_abs_max": float(selected.abs().max()) if selected.numel() else None,
    }


def preserve_strides(value, dtype):
    result = torch.empty_strided(value.shape, value.stride(), dtype=dtype, device=value.device)
    result.copy_(value.detach())
    return result


def attention_result(qkv, incoming, backend, dtype):
    """Return identical-input forward/backward results without any parameter update."""
    context = sdpa_kernel(backend) if backend is not None else sdpa_kernel([
        SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION])
    with context:
        q, k, v = [preserve_strides(value, dtype).requires_grad_(True) for value in qkv]
        grad_out = preserve_strides(incoming, dtype)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        implementation = type(output.grad_fn).__name__
        gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=grad_out)
    return {"implementation": implementation, "output": output.detach(),
            "gradients": tuple(value.detach() for value in gradients)}


def compare_with_reference(result, reference):
    data = {"implementation": result["implementation"], "forward": summary(result["output"]),
            "forward_abs_error": finite_number((result["output"].double()-reference["output"]).abs().max()),
            "gradients": {}}
    for name, value, expected in zip(("q", "k", "v"), result["gradients"], reference["gradients"]):
        error = float((value.double()-expected).abs().max())
        reference_max = float(expected.abs().max())
        data["gradients"][name] = {
            **summary(value), "reference_abs_max": reference_max,
            "absolute_error": finite_number(error),
            "error_relative_to_reference_max": finite_number(error / max(reference_max, 1e-300)),
        }
    return data


def capture_first_attention(model, values, ids, targets, divisor):
    """Use the math backend for the model path; isolate only its first attention call."""
    captured = {}
    original = F.scaled_dot_product_attention

    def capture(q, k, v, *args, **kwargs):
        output = original(q, k, v, *args, **kwargs)
        if not captured:
            captured["qkv"] = tuple(value.detach().clone() for value in (q, k, v))
            def incoming(gradient):
                captured["incoming"] = gradient.detach().clone()
                return gradient
            output.register_hook(incoming)
        return output

    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH), patch.object(F, "scaled_dot_product_attention", capture):
        predictions = model(values)
        loss, selected = training.compute_masked_loss(predictions, ids, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Full-model MATH FP32 evaluation loss is nonfinite.")
        (loss / divisor).backward()
    if "incoming" not in captured:
        raise RuntimeError("Could not capture the first SDPA output gradient with this training script.")
    for tensor in (*captured["qkv"], captured["incoming"]):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("Full-model MATH FP32 probe already produced invalid attention tensors/gradients.")
    captured["loss"] = float(loss.detach())
    captured["prediction"] = float(selected.detach()[0])
    return captured


def run_probe(args):
    payload = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if payload.get("diagnostic_format") != 1 or payload.get("snapshot_phase") != "after_adam_step":
        raise ValueError("Expected diagnose_nonfinite.py's after_adam_step first_failure_state.pt.")
    previous_report = payload["report"]
    model_state = payload["model_state_dict"]
    del payload
    allowed = {field.name for field in fields(training.Config)}
    values = dict(previous_report["config"])
    if set(values) - allowed:
        raise ValueError("Snapshot configuration differs from this training script.")
    values["input"] = args.input
    for key in ("head_hidden", "excluded_features"):
        values[key] = tuple(values[key])
    config = training.Config(**values)
    config.validate()
    vocabulary = training.build_vocabulary(config.input, config.excluded_features)
    if vocabulary.sha256 != previous_report["vocabulary_sha256"]:
        raise ValueError("Input vocabulary differs from the diagnostic snapshot.")
    profiles, audit = training.load_profiles(config.input, vocabulary)
    if audit["data_sha256"] != previous_report["data_sha256"]:
        raise ValueError("Input retained profiles differ from the diagnostic snapshot.")
    profile = next((p for p in profiles if p.run_id == args.run_id), None)
    if profile is None:
        raise ValueError(f"Run {args.run_id} is absent from the input.")
    if max(1, math.floor(config.mask_fraction*len(profile.species_ids))) != 1:
        raise ValueError("This focused probe supports runs whose configured masking selects one entry.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use the original GPU environment.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, or cuda:N.")
    torch.set_num_threads(config.threads)
    model = training.build_model(config, vocabulary.size, device)
    model.load_state_dict(model_state, strict=True)
    del model_state
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"Snapshot model parameter {name} is nonfinite; this probe requires finite weights.")
    out = Path(args.output_dir or ("attention_probe_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    snapshot = Path(args.snapshot).resolve()
    if out == snapshot.parent or snapshot.parent in out.parents:
        raise ValueError("Choose a new output directory outside the existing diagnostic folder.")
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "snapshot": str(snapshot), "weights_from": "after failed Adam update",
        "run_id": profile.run_id, "reported_entries": len(profile.species_ids),
        "dropout_for_this_probe": 0.0, "reference": "isolated MATH FP64 attention",
        "limitation": "Post-update snapshot; dropout disabled for controlled comparison. "
                      "This does not exactly recreate the original training-mode failure. "
                      "A clean probe cannot rule out that failure.",
        "cases": [],
    }
    result_path = out / "attention_comparison.json"
    print("FOCUSED ATTENTION PROBE; evaluation mode; no optimizer updates", flush=True)
    print(f"Run {profile.run_id}; testing all {len(profile.species_ids)} one-entry masks", flush=True)
    print(f"Report: {result_path}", flush=True)
    for species_id in profile.species_ids:
        species_id = int(species_id)
        masked = training.mask_profile(profile, config.mask_fraction, None, [species_id])
        case = {"masked_species_id": species_id, "masked_label": vocabulary.labels[species_id-1],
                "target_text": masked.target_text[0]}
        try:
            values, ids, targets = training.prepare_batch(masked, device)
            captured = capture_first_attention(model, values, ids, targets,
                                                previous_report["context"]["group_size"])
            qkv, incoming = captured["qkv"], captured["incoming"]
            case.update(eval_loss=captured["loss"], prediction=captured["prediction"],
                        inputs={name: summary(tensor) for name, tensor in zip(("q", "k", "v"), qkv)},
                        incoming_gradient=summary(incoming))
            scores = qkv[0].double() @ qkv[1].double().transpose(-2,-1) / math.sqrt(qkv[0].shape[-1])
            case["attention_score_range_float64"] = [float(scores.min()), float(scores.max())]
            reference = attention_result(qkv, incoming, SDPBackend.MATH, torch.float64)
            case["backends"] = {}
            for label, backend in (("math_fp32", SDPBackend.MATH),
                                   ("default_fp32", None),
                                   ("efficient_fp32", SDPBackend.EFFICIENT_ATTENTION)):
                try:
                    result = attention_result(qkv, incoming, backend, torch.float32)
                    case["backends"][label] = compare_with_reference(result, reference)
                    gradients = case["backends"][label]["gradients"]
                    max_error = max(float(entry["absolute_error"]) for entry in gradients.values())
                    max_gradient = max(float(entry["finite_abs_max"] or 0) for entry in gradients.values())
                    invalid = sum(entry[key] for entry in gradients.values()
                                  for key in ("nan", "positive_inf", "negative_inf"))
                    print(f"mask ID {species_id}; {label}; {result['implementation']}; "
                          f"max|gradient|={max_gradient:.6g}; max absolute gradient error={max_error:.6g}; "
                          f"nonfinite gradient entries={invalid}", flush=True)
                except RuntimeError as error:
                    case["backends"][label] = {"unavailable_or_error": str(error)}
                    print(f"mask ID {species_id}; {label}: unavailable/error: {error}", flush=True)
        except Exception as error:
            case["probe_error"] = f"{type(error).__name__}: {error}"
            print(f"mask ID {species_id}: {case['probe_error']}", flush=True)
        report["cases"].append(case)
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Probe finished. This probe does not resume or repair training. Report: {result_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", default="nonfinite_replay/first_failure_state.pt")
    parser.add_argument("--input", default="non_asd_qc_pass_reported_required.tsv.gz")
    parser.add_argument("--run-id", default="SRR25256440")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", help="A new directory; defaults to a timestamped name.")
    args = parser.parse_args()
    try:
        run_probe(args)
    except Exception as error:
        print(f"ATTENTION PROBE ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
