import os, random
from pathlib import Path
import torch


def rng_state():
    return dict(
        python=random.getstate(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["cpu"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, state, meta):
    atomic_save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            state=state,
            rng=rng_state(),
            meta=meta,
        ),
        path,
    )


def load_checkpoint(path, model, optimizer, meta):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["meta"] != meta:
        raise ValueError(
            "Checkpoint/config/input/code mismatch. Resume requires identical experiment metadata."
        )
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    return saved["state"]
