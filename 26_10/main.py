import argparse, hashlib, json, platform
from pathlib import Path
import torch
from config import load_config
from data import load_data
from split import make_split
from masking import choose_mask
from model import MicrobiomeEncoder
from runtime import seed_all
from checkpoints import save_checkpoint, load_checkpoint
from diagnostics import Monitor
from train import train_epoch
from validate import validate
from outputs import write_json, update_history, plot_results


def experiment_meta(c, vocabulary, digest, device):
    code = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    return dict(
        config=c,
        vocabulary=vocabulary,
        input_sha256=digest,
        code=code,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        python=platform.python_version(),
    )


def setup(c, device):
    runs, vocabulary, digest = load_data(c["input"], c["excluded_features"])
    train_ids, val_ids = make_split(len(runs), c["validation_fraction"], c["seed"])
    masks = {
        i: choose_mask(
            len(runs[i]["ids"]),
            c["mask_fraction"],
            torch.Generator().manual_seed(c["seed"] + 900000007 + i),
        )
        for i in val_ids
    }
    model = MicrobiomeEncoder(c, len(vocabulary)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(c["beta1"], c["beta2"]), eps=c["adam_eps"]
    )
    return (
        runs,
        vocabulary,
        train_ids,
        val_ids,
        masks,
        model,
        optimizer,
        experiment_meta(c, vocabulary, digest, device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("settings.json"))
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    args = parser.parse_args()
    c = load_config(args.config)
    seed_all(c["seed"])
    device = torch.device(args.device)
    out = Path(c["output"]).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError(
            "Output directory is not empty. Choose a new output or explicitly resume."
        )
    out.mkdir(parents=True, exist_ok=True)
    runs, vocab, tr, va, masks, model, opt, meta = setup(c, device)
    state = dict(epoch=1, update=0, order=None, position=0, total_loss=0.0, count=0)
    if args.resume:
        state = load_checkpoint(args.resume, model, opt, meta)
    write_json(
        out / "experiment.json",
        dict(
            **meta,
            split=dict(
                train=[runs[i]["run_id"] for i in tr],
                validation=[runs[i]["run_id"] for i in va],
                validation_masks={runs[i]["run_id"]: masks[i].tolist() for i in va},
                donor_grouping=False,
            ),
        ),
    )
    monitor = Monitor(out / "diagnostic_events.jsonl")
    print(
        f'Runs {len(runs):,}; vocabulary {len(vocab):,}; train {len(tr):,}; validation {len(va):,}; FP32; automatic attention={c["attention_backend"]=="auto"}; no clipping',
        flush=True,
    )
    for epoch in range(state["epoch"], c["epochs"] + 1):
        state["epoch"] = epoch
        train_mse = train_epoch(
            model, opt, runs, tr, c, device, state, meta, out, monitor
        )
        val_mse, pairs = validate(model, runs, va, masks, device, c)
        plotting = epoch % c["plot_every"] == 0 or epoch == c["epochs"]
        history = update_history(
            out, epoch, train_mse, val_mse, pairs if plotting else None
        )
        state.update(epoch=epoch + 1, order=None, position=0, total_loss=0.0, count=0)
        if epoch % c["checkpoint_every"] == 0 or epoch == c["epochs"]:
            save_checkpoint(
                out / "checkpoints" / f"epoch_{epoch:04d}.pt", model, opt, state, meta
            )
        if epoch % c["plot_every"] == 0 or epoch == c["epochs"]:
            plot_results(out, epoch, history)
        print(
            f"Completed epoch {epoch}: train MSE={train_mse:.7g}; validation MSE={val_mse:.7g}",
            flush=True,
        )
    print("Completed. Results:", out, flush=True)


if __name__ == "__main__":
    main()
