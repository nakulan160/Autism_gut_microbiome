#!/usr/bin/env python3
"""Compare gradient clipping on CPU, using copies of saved gradient tensors.

Run:
    python -u compare_gradient_clipping_cpu.py --gradients failure_gradients_only.pt

Methods:
1. The unchanged custom float64-norm clipping function from the guarded trainer.
2. torch.nn.utils.clip_grad_norm_ on the original float32 gradients.
3. The same PyTorch function on float64 copies, then cast back to float32.

The third method is a numerical reference, not a change to the training code.
Two cases are compared: the actual saved gradients, and a clearly labelled
rescaled copy with global norm about 10. The second case is artificial and
shows how the methods compare when the norm calculation does not overflow.

No forward/backward pass, optimizer step, or training checkpoint write occurs.
All tensors stay on CPU. Only PyTorch and Python's standard library are needed.

Official references checked when preparing this script:
https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html
https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html
https://docs.pytorch.org/docs/2.8/generated/torch.load.html
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


# These two functions are copied unchanged from the guarded trainer.
def nonfinite_tensor_names(named_tensors) -> list[str]:
    groups = {}
    for name, value in named_tensors:
        if torch.is_tensor(value):
            groups.setdefault(value.device, []).append((name, value.detach()))
    bad = []
    for entries in groups.values():
        flags = torch.stack([torch.isfinite(value).all() for _, value in entries]).cpu().tolist()
        bad.extend(name for (name, _), finite in zip(entries, flags) if not finite)
    return bad


@torch.no_grad()
def clip_gradients_safely(model, max_norm: float) -> dict:
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if not gradients:
        raise FloatingPointError("No gradients were produced for this optimizer update.")
    # Float64 handles the square of any finite float32 gradient without overflow.
    norms = torch.stack([torch.linalg.vector_norm(gradient.detach(), dtype=torch.float64)
                         for _, gradient in gradients])
    norm = float(torch.linalg.vector_norm(norms))
    if not math.isfinite(norm):
        bad = nonfinite_tensor_names(gradients)
        raise FloatingPointError("Nonfinite gradient norm before Adam; affected tensors: "
                                 + ", ".join(bad[:12]))
    scale = min(1.0, max_norm / (norm + 1e-6))
    clipped = norm > max_norm
    if clipped:
        for _, gradient in gradients:
            # Do not cast a tiny scaling coefficient to float32 before multiplying.
            gradient.copy_((gradient.detach().double() * scale).to(gradient.dtype))
        after = float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(gradient.detach(), dtype=torch.float64)
            for _, gradient in gradients])))
        if not math.isfinite(after) or after > max_norm * (1 + 1e-6):
            raise FloatingPointError("Gradient clipping did not produce the requested finite norm.")
    else:
        after = norm
        scale = 1.0
    return {"norm_before": norm, "norm_after": after, "scale": scale, "clipped": clipped}


def load_gradients(path):
    """Load the tensor-only extract without restoring a model or optimizer."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    gradients = state.get("gradients") if isinstance(state, dict) else None
    if not isinstance(gradients, dict) or not gradients:
        raise ValueError("Expected a non-empty dictionary under the key 'gradients'.")
    for name, gradient in gradients.items():
        if not isinstance(name, str) or not isinstance(gradient, torch.Tensor):
            raise ValueError("Expected parameter names mapped to gradient tensors.")
        if gradient.layout != torch.strided or gradient.dtype != torch.float32:
            raise ValueError(f"{name}: this comparison expects dense float32 gradients.")
        if gradient.numel() == 0 or not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"{name}: gradient is empty or contains NaN/Inf.")
    return {name: gradient.detach() for name, gradient in gradients.items()}


def global_l2_float64(gradients):
    """Measure the global norm without squaring large values in float32."""
    norms = torch.stack([
        torch.linalg.vector_norm(gradient.detach(), dtype=torch.float64)
        for gradient in gradients
    ])
    return float(torch.linalg.vector_norm(norms))


def make_parameter_copies(gradients, dtype):
    """Give the library function real Parameters with independent .grad copies."""
    model = nn.ParameterList()
    for gradient in gradients.values():
        copied_gradient = gradient.to(device="cpu", dtype=dtype, copy=True)
        parameter = nn.Parameter(torch.empty_like(copied_gradient))
        parameter.grad = copied_gradient
        model.append(parameter)
    return model


def compare_values(reference, candidate):
    """Compare all gradient elements, with float64 arithmetic for the metrics."""
    squared_error = 0.0
    squared_reference = 0.0
    largest_difference = 0.0
    bitwise_equal = True
    close = True
    for expected, actual in zip(reference, candidate):
        expected64 = expected.double()
        difference = actual.double() - expected64
        squared_error += float(torch.sum(difference.square()))
        squared_reference += float(torch.sum(expected64.square()))
        largest_difference = max(largest_difference, float(difference.abs().max()))
        bitwise_equal = bitwise_equal and torch.equal(expected, actual)
        close = close and torch.allclose(expected, actual, rtol=1e-5, atol=1e-8)
    relative_l2 = (
        math.sqrt(squared_error / squared_reference)
        if squared_reference > 0.0 else None
    )
    return {
        "max_absolute_difference": largest_difference,
        "relative_l2_difference": relative_l2,
        "bitwise_equal": bitwise_equal,
        "allclose_rtol_1e-5_atol_1e-8": close,
    }


@torch.no_grad()
def run_method(gradients, method, max_norm):
    """Run one method on fresh copies. An expected clipping error is recorded."""
    dtype = torch.float64 if method == "pytorch_float64_reference" else torch.float32
    model = make_parameter_copies(gradients, dtype)
    summary = {"method": method, "working_dtype": str(dtype)}
    try:
        if method == "custom_float64_norm":
            summary["custom_details"] = clip_gradients_safely(model, max_norm)
        else:
            # This is the official built-in, not a reimplementation.
            returned_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=None,
            )
            summary["returned_norm_before_clipping"] = float(returned_norm)

        # The training model stores float32 gradients. Compare in that dtype.
        clipped = [parameter.grad.detach().to(dtype=torch.float32) for parameter in model]
        norm_after = global_l2_float64(clipped)
        finite = all(bool(torch.isfinite(value).all()) for value in clipped)
        within_threshold = finite and norm_after <= max_norm * (1.0 + 1e-6)
        summary.update({
            "status": "ok" if within_threshold else "invalid_output",
            "all_output_values_finite": finite,
            "norm_after_in_float32": norm_after,
            "within_threshold_allowing_roundoff": within_threshold,
        })
        if not within_threshold:
            return summary, None
        return summary, clipped
    except (RuntimeError, FloatingPointError) as error:
        summary.update({"status": "error", "error": str(error)})
        # With error_if_nonfinite=True, the built-in may reject an overflowing
        # norm even when every input gradient value is finite. Keep that result.
        return summary, None


def run_case(name, gradients, max_norm, description):
    print(f"\nCASE: {name}", flush=True)
    print(description, flush=True)
    norm_before = global_l2_float64(gradients.values())
    print(f"Global L2 norm before clipping (float64): {norm_before:.9g}", flush=True)
    reference_summary, reference = run_method(gradients, "custom_float64_norm", max_norm)
    results = [reference_summary]
    if reference is None:
        print(f"custom_float64_norm: {reference_summary['status']}", flush=True)
    else:
        print(
            "custom_float64_norm: OK; "
            f"norm after = {reference_summary['norm_after_in_float32']:.9g}",
            flush=True,
        )

    for method in ("pytorch_float32", "pytorch_float64_reference"):
        summary, clipped = run_method(gradients, method, max_norm)
        if clipped is not None:
            if reference is not None:
                summary["comparison_with_custom"] = compare_values(reference, clipped)
            print(
                f"{method}: OK; norm after = {summary['norm_after_in_float32']:.9g}",
                flush=True,
            )
            comparison = summary.get("comparison_with_custom")
            if comparison:
                print(
                    f"  max difference = {comparison['max_absolute_difference']:.9g}; "
                    f"relative L2 difference = {comparison['relative_l2_difference']}; "
                    f"allclose = {comparison['allclose_rtol_1e-5_atol_1e-8']}",
                    flush=True,
                )
        else:
            print(f"{method}: {summary['status'].upper()}", flush=True)
            if "error" in summary:
                print(f"  {summary['error']}", flush=True)
        results.append(summary)
        del clipped
    return {
        "case": name,
        "description": description,
        "global_l2_before_float64": norm_before,
        "methods": results,
    }


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("Must be a finite number greater than zero.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--max-norm", type=positive_float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("clipping_comparison.json"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    torch.set_num_threads(args.threads)

    print(f"PyTorch: {torch.__version__}; device: CPU", flush=True)
    gradients = load_gradients(args.gradients)
    norm_before = global_l2_float64(gradients.values())
    largest_name, largest_gradient = max(
        gradients.items(), key=lambda item: float(item[1].abs().max())
    )
    print(
        f"Loaded {len(gradients):,} gradient tensors; "
        f"{sum(value.numel() for value in gradients.values()):,} elements.",
        flush=True,
    )
    print(f"Largest |gradient|: {float(largest_gradient.abs().max()):.9g}", flush=True)
    print(f"Parameter containing it: {largest_name}", flush=True)
    print(f"Clipping threshold for every method: {args.max_norm}", flush=True)

    report = {
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "input": str(args.gradients.resolve()),
        "gradient_tensor_count": len(gradients),
        "gradient_element_count": sum(value.numel() for value in gradients.values()),
        "all_input_values_finite": True,
        "largest_absolute_gradient": float(largest_gradient.abs().max()),
        "largest_gradient_parameter": largest_name,
        "max_norm": args.max_norm,
        "cases": [],
    }
    report["cases"].append(run_case(
        "actual_saved_gradients", gradients, args.max_norm,
        "Original recorded gradients; values have not been rescaled.",
    ))

    # An explicit control using the same direction at an ordinary magnitude.
    # It is not a newly measured training gradient.
    if norm_before > 0.0:
        factor = (10.0 * args.max_norm) / norm_before
        rescaled = {
            name: (value.double() * factor).to(dtype=value.dtype)
            for name, value in gradients.items()
        }
        control = run_case(
            "artificially_rescaled_copy", rescaled, args.max_norm,
            "Same saved gradient direction, rescaled to a global norm near "
            f"{10.0 * args.max_norm:g}. This is an artificial control.",
        )
        control["rescaling_factor"] = factor
        report["cases"].append(control)

    report["limits"] = [
        "This compares clipping arithmetic at one recorded update.",
        "It does not establish why the original gradient spike happened.",
        "It does not predict the result of a new 100-epoch training run.",
        "The float64 built-in case uses copies for comparison only.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"\nSaved report: {args.output.resolve()}", flush=True)
    print("An error in the float32 method is recorded as a comparison result.", flush=True)
    print("Share clipping_comparison.json so we can interpret your CPU result.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(f"COMPARISON ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

