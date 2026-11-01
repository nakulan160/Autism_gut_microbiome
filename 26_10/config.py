import json
from pathlib import Path


def load_config(path):
    """Read explicit experiment settings; reject unknown keys and invalid combinations."""
    defaults = json.loads(Path(__file__).with_name("settings.json").read_text())
    supplied = json.loads(Path(path).read_text())
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    defaults.update(supplied)
    c = defaults
    for k in [
        "epochs",
        "species_dim",
        "abundance_dim",
        "layers",
        "heads",
        "feedforward",
        "accumulation",
        "warmup_updates",
        "checkpoint_every",
        "plot_every",
        "recovery_every_updates",
    ]:
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(k)
    if not 0 < c["validation_fraction"] < 1 or not 0 < c["mask_fraction"] <= 1:
        raise ValueError("Invalid fraction")
    if (c["species_dim"] + c["abundance_dim"]) % c["heads"]:
        raise ValueError("Combined dimensions must divide by heads")
    if c["attention_backend"] not in ["auto", "math"]:
        raise ValueError("attention_backend")
    if not 0 <= c["dropout"] < 1:
        raise ValueError("dropout")
    for k in ["beta1", "beta2"]:
        if not 0 <= c[k] < 1:
            raise ValueError(k)
    if c["adam_eps"] <= 0 or c["gradient_warning"] <= 0:
        raise ValueError("Invalid diagnostic/Adam setting")
    if not c["head_dims"] or any(
        not isinstance(x, int) or x < 1 for x in c["head_dims"]
    ):
        raise ValueError("head_dims")
    return c
