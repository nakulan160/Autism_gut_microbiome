"""Resume the unchanged modular trainer with ONLY the finite-gradient magnitude stop disabled.
Place beside main.py. Preserve all original source files and settings.json.
Training: python resume_nonfinite_only.py --config settings.json --device cuda --resume results_modular/recovery.pt
Later replay: python resume_nonfinite_only.py --replay --results results_modular --device cuda
"""
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import main as trainer
import train
from checkpoints import restore_rng

POLICY = {"finite_gradient_limit": None, "stop_on_nonfinite": True, "gradient_clipping": False}
_original_meta = trainer.experiment_meta
_original_epoch = trainer.train_epoch


def experiment_meta(*args, **kwargs):
    meta = _original_meta(*args, **kwargs)
    meta["code"].pop(Path(__file__).name, None)
    meta["execution_policy"] = dict(POLICY)
    meta["policy_launcher_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return meta


def base_meta(meta):
    meta = dict(meta)
    meta.pop("execution_policy", None)
    meta.pop("policy_launcher_sha256", None)
    return meta


def load_checkpoint(path, model, optimizer, meta):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    previous_policy = saved["meta"].get("execution_policy")
    if previous_policy is not None and previous_policy != POLICY:
        raise ValueError("Checkpoint has an unsupported execution policy")
    if base_meta(saved["meta"]) != base_meta(meta):
        raise ValueError("Original configuration, code, input or environment differs. Restore the original project files/settings before resuming.")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    print("Resuming saved state. Finite-gradient magnitude stop DISABLED; NaN/Inf checks remain enabled; no clipping.", flush=True)
    return saved["state"]


def train_epoch(model, optimizer, runs, indices, config, *args, **kwargs):
    runtime_config = dict(config)
    # Original trainer compares max_abs > gradient_warning. Every finite FP32
    # value is below infinity. Existing finite checks run first and remain intact.
    runtime_config["gradient_warning"] = float("inf")
    return _original_epoch(model, optimizer, runs, indices, runtime_config, *args, **kwargs)


def preserve_diagnostics(checkpoint):
    folder = Path(checkpoint).resolve().parent
    if not (folder / "failure.pt").exists():
        return
    destination = folder / ("preserved_failure_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    destination.mkdir()
    for name in ["failure.pt", "replay_trace.jsonl", "recovery.pt", "experiment.json"]:
        source = folder / name
        if not source.exists():
            continue
        # A hard link keeps the old recovery state when the rolling file is
        # atomically replaced, avoiding a large temporary copy where supported.
        if name == "recovery.pt":
            try:
                (destination / name).hardlink_to(source)
            except OSError:
                shutil.copy2(source, destination / name)
        else:
            shutil.copy2(source, destination / name)
    print("Previous diagnostic evidence preserved in:", destination, flush=True)


def run():
    trainer.experiment_meta = experiment_meta
    trainer.load_checkpoint = load_checkpoint
    trainer.train_epoch = train_epoch
    if "--replay" in sys.argv:
        sys.argv.remove("--replay")
        import replay
        replay.main()
        return
    if "--resume" not in sys.argv:
        raise SystemExit("Supply --resume results_modular/recovery.pt; this launcher is for continuing saved training.")
    index = sys.argv.index("--resume")
    if index + 1 >= len(sys.argv):
        raise SystemExit("Missing checkpoint path")
    preserve_diagnostics(sys.argv[index + 1])
    trainer.main()


if __name__ == "__main__":
    run()
