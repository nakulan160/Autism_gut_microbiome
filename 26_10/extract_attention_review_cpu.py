#!/usr/bin/env python3
"""Package saved FP32 attention evidence and the COMPLETE trace for review on CPU.

Input: the three files produced by trace_gradient_explosion_fp32.py (format 2).
Output: one ZIP containing attention_only.pt, operation_trace.tsv, and report.txt.
The original files are opened read-only. This script never runs the model.
"""

import argparse
import copy
import io
import os
import pickle
import re
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import torch
import numpy as np


TENSOR_KEY = re.compile(r"tensor_\d+\Z")
CASE_FIELDS = (
    "context", "sample", "snapshot_phase", "weights_note", "status",
    "analysis_error", "analysis_traceback", "original_spike", "nodes_pass_name",
    "same_graph_reproduction", "torch_rng_unchanged_by_repeat",
    "first_attention_parameter_gradient_heads",
    "first_attention_parameter_gradient_heads_source",
    "observed_gradient_growth_boundaries", "boundary_note",
    "native_saved_scalar_dtype_note", "accumulation_group_before_this_sample",
    "accumulation_group_complete", "adam_learning_rate", "adam_result",
    "fp32_adam_square_observations", "rng_before_sample_forward",
    "rng_after_original_backward", "forward_tensors", "original_forward_tensors",
)
NODE_PASSES = ("nodes", "original_backward_nodes", "retained_graph_repeat_nodes")
PARAMETER_GROUPS = (
    "individual_gradients_original_backward",
    "actual_accumulated_gradients_before_adam",
)


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read this diagnostic's NumPy 2 arrays/scalars with NumPy 1.x too.

    NumPy 2 pickles refer to numpy._core. Older NumPy stores the equivalent
    ndarray/scalar reconstruction functions under numpy.core. This maps their
    module names at deserialization time, without changing array contents or
    aliasing modules globally. All unrelated pickle references are unchanged.
    """

    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2:
            if module == "numpy._core" or module.startswith("numpy._core."):
                module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def numpy_compat_load(file, **kwargs):
    return NumpyCompatUnpickler(file, **kwargs).load()


def numpy_compat_loads(data, **kwargs):
    return numpy_compat_load(io.BytesIO(data), **kwargs)


COMPAT_PICKLE = types.ModuleType("attention_numpy_compat_pickle")
COMPAT_PICKLE.Unpickler = NumpyCompatUnpickler
COMPAT_PICKLE.Pickler = pickle.Pickler
COMPAT_PICKLE.load = numpy_compat_load
COMPAT_PICKLE.loads = numpy_compat_loads
COMPAT_PICKLE.dump = pickle.dump
COMPAT_PICKLE.dumps = pickle.dumps


def tensor_references(value):
    """Find the archive references in the original diagnostic's encoded records."""
    if isinstance(value, str) and TENSOR_KEY.fullmatch(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_references(item)


def clone_cpu(value):
    """Preserve tensor values/dtypes, without saving a larger shared storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def is_attention(record):
    name = record.get("operation", "")
    return "ScaledDotProduct" in name and "Backward" in name


def compact_case(case):
    """Retain attention from ALL heads/layers and original/repeated backwards."""
    kept = {key: clone_cpu(case[key]) for key in CASE_FIELDS if key in case}
    found_first_attention = False
    for field in NODE_PASSES:
        if field not in case:
            continue
        selected = [entry for entry in case[field] if is_attention(entry)]
        kept[field] = clone_cpu(selected)
        found_first_attention |= any(
            entry.get("scope") == "blocks.0.self_attn" for entry in selected
        )
    if not found_first_attention:
        raise ValueError("The captured case has no first-encoder native attention node.")

    checks = case.get("fp32_algebra_checks", [])
    kept["fp32_algebra_checks"] = clone_cpu([x for x in checks if is_attention(x)])
    # Retain comparison numbers for other operators without their large matrices.
    kept["other_operator_check_summaries"] = [
        clone_cpu({key: item for key, item in check.items()
                   if not tuple(tensor_references(item))})
        for check in checks if not is_attention(check)
    ]

    # Keep the original first-attention/embedding parameter gradients as evidence.
    for field in PARAMETER_GROUPS:
        if field in case:
            kept[field] = clone_cpu({
                name: tensor for name, tensor in case[field].items()
                if name.startswith(("embedding.", "blocks.0.self_attn."))
            })

    archive = case.get("archive", {})
    original_tensors = archive.get("tensors", {})
    needed = set(tensor_references(kept))
    missing = sorted(needed.difference(original_tensors))
    if missing:
        raise ValueError("Archive tensor references missing: " + ", ".join(missing[:10]))
    if not needed:
        raise ValueError("No attention tensor references were found; check the bundle format.")
    kept["archive"] = {
        "tensors": {key: clone_cpu(original_tensors[key]) for key in sorted(needed)},
        "statistics": clone_cpu({
            key: archive.get("statistics", {})[key]
            for key in sorted(needed) if key in archive.get("statistics", {})
        }),
    }
    kept["extraction_note"] = (
        "All attention heads/layers, their saved native operands and backward tensors, "
        "and all recorded module forward tensors are retained. Tensor values/dtypes "
        "are unchanged; extracted storage is contiguous CPU storage. Original native "
        "strides/device metadata remain in archive statistics. Full model/Adam states "
        "and other parameters' gradient arrays remain in the original bundle. "
        "The complete operation_trace.tsv accompanies this extract in the ZIP."
    )
    return kept


def make_package(folder, output):
    source = folder / "diagnostic_bundle.pt"
    trace = folder / "operation_trace.tsv"
    report = folder / "report.txt"
    for path in (source, trace, report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Already exists: {output}. Choose a new --output filename.")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Opening saved diagnostic bundle on CPU...", flush=True)
    print("Local versions: PyTorch", torch.__version__, "; NumPy", np.__version__, flush=True)
    if int(np.__version__.split(".")[0]) < 2:
        print("Reading NumPy 2 module names using the installed NumPy 1.x equivalents.", flush=True)
    # Ordinary CPU loading also supports older torch.load versions without mmap.
    # This reads tensor storages into RAM; values and dtypes are not changed.
    # The source is the user's own diagnostic, including its saved RNG objects.
    bundle = torch.load(source, map_location="cpu", weights_only=False,
                        pickle_module=COMPAT_PICKLE)
    if bundle.get("diagnostic_format") != 2:
        raise ValueError("Expected diagnostic_format=2 from trace_gradient_explosion_fp32.py.")
    cases = bundle.get("cases", [])
    if not cases:
        raise ValueError("The source bundle has no captured diagnostic cases.")
    compact = {
        "attention_extract_format": 1,
        "source_diagnostic_format": bundle["diagnostic_format"],
        "source_bundle_bytes": source.stat().st_size,
        "extraction_environment": {
            "torch": str(torch.__version__), "numpy": np.__version__,
            "numpy_core_name_mapping": int(np.__version__.split(".")[0]) < 2,
            "loading": "ordinary CPU loading without mmap",
        },
        "report": clone_cpu(bundle.get("report", {})),
        "vocabulary_labels": clone_cpu(bundle.get("vocabulary_labels", [])),
        "original_trainer_source": bundle.get("original_trainer_source", ""),
        "script_source": bundle.get("script_source", ""),
        "extractor_source": Path(__file__).read_text(encoding="utf-8"),
        "cases": [],
    }
    for case in cases:
        extracted = compact_case(case)
        compact["cases"].append(extracted)
        print("Captured:", extracted.get("context", {}),
              "retained tensors:", len(extracted["archive"]["tensors"]), flush=True)

    with tempfile.TemporaryDirectory(prefix="attention_extract_", dir=output.parent) as temporary:
        temporary = Path(temporary)
        small_pt = temporary / "attention_only.pt"
        torch.save(compact, small_pt)
        packed = temporary / "attention_review.zip"
        print("Compressing attention data and the COMPLETE trace table...", flush=True)
        with zipfile.ZipFile(packed, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            archive.write(small_pt, "attention_only.pt")
            archive.write(trace, "operation_trace.tsv")
            archive.write(report, "report.txt")
        # Atomic, non-overwriting publication on the same local filesystem.
        os.link(packed, output)
    print("Saved:", output, flush=True)
    print(f"Original bundle: {source.stat().st_size / 1024**2:,.1f} MiB", flush=True)
    print(f"ZIP for upload: {output.stat().st_size / 1024**2:,.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.input_dir.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output
              else folder / "attention_review.zip")
    make_package(folder, output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"EXTRACTION ERROR: {error}", file=sys.stderr)
        sys.exit(1)
