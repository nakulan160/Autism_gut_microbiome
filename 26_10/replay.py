"""Recreate the flagged sample group from a rotating recovery checkpoint; never update source files."""

import argparse
from pathlib import Path
import torch
from main import setup
from runtime import seed_all, attention_context, learning_rate
from checkpoints import restore_rng
from masking import choose_mask, prepare_sample
from losses import masked_loss
from diagnostics import Trace, gradient_summary, check_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    folder = Path(args.results)
    failure = torch.load(folder / "failure.pt", map_location="cpu", weights_only=False)
    saved = torch.load(folder / "recovery.pt", map_location="cpu", weights_only=False)
    c = saved["meta"]["config"]
    seed_all(c["seed"])
    device = torch.device(args.device)
    runs, vocab, tr, va, masks, model, opt, meta = setup(c, device)
    if meta != saved["meta"]:
        raise ValueError(
            "Replay environment/config/input/code differs from recorded experiment"
        )
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    model.train()
    state = saved["state"]
    target_update = failure["records"][-1]["update"]
    epoch = state["epoch"]
    if state["order"] is None:
        g = torch.Generator().manual_seed(c["seed"] + epoch)
        state["order"] = [tr[j] for j in torch.randperm(len(tr), generator=g).tolist()]
    destination = folder / "replay_trace.jsonl"
    if destination.exists():
        raise ValueError("replay_trace.jsonl exists; rename it before another replay")
    for start in range(state["position"], len(state["order"]), c["accumulation"]):
        group = state["order"][start : start + c["accumulation"]]
        opt.zero_grad(set_to_none=True)
        update = state["update"] + 1
        with attention_context(c["attention_backend"]):
            for i in group:
                mask = choose_mask(
                    len(runs[i]["ids"]),
                    c["mask_fraction"],
                    torch.Generator().manual_seed(c["seed"] + epoch * 1000003 + i),
                )
                ids, a, ti, t = prepare_sample(runs[i], mask, device)
                if update == target_update:
                    with open(destination, "a") as f:
                        import json

                        f.write(
                            json.dumps(
                                dict(
                                    run_id=runs[i]["run_id"],
                                    visible_species_ids=ids.cpu().tolist(),
                                    masked_species_ids=ti.cpu().tolist(),
                                    targets=t.cpu().tolist(),
                                )
                            )
                            + "\n"
                        )
                    with Trace(model, destination) as trace:
                        loss = masked_loss(model(ids, a), ti, t) / len(group)
                        trace.graph(loss)
                        loss.backward()
                    print(runs[i]["run_id"], gradient_summary(model), flush=True)
                else:
                    (masked_loss(model(ids, a), ti, t) / len(group)).backward()
        if update == target_update:
            print(
                "Trace complete before optimizer step. Fused attention internals may require a separate instrumented backend comparison."
            )
            return
        for pg in opt.param_groups:
            pg["lr"] = learning_rate(update, c)
        opt.step()
        check_state(model, opt)
        state["update"] = update
    raise RuntimeError("Target update not reached")


if __name__ == "__main__":
    main()
