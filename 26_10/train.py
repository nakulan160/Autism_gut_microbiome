import math, torch
from pathlib import Path
from masking import choose_mask, prepare_sample
from losses import masked_loss
from runtime import learning_rate, attention_context
from diagnostics import check_tensor, gradient_summary, check_state, DiagnosticStop
from checkpoints import save_checkpoint, atomic_save


def train_epoch(
    model, optimizer, runs, train_indices, c, device, state, meta, out, monitor
):
    """One sample per forward; average gradients over accumulation groups (last group uses actual size)."""
    model.train()
    epoch = state["epoch"]
    if state.get("order") is None:
        generator = torch.Generator().manual_seed(c["seed"] + epoch)
        state.update(
            order=[
                train_indices[j]
                for j in torch.randperm(
                    len(train_indices), generator=generator
                ).tolist()
            ],
            position=0,
            total_loss=0.0,
            count=0,
        )
    order = state["order"]
    start = state["position"]
    for group_start in range(start, len(order), c["accumulation"]):
        if group_start == 0 or state["update"] % c["recovery_every_updates"] == 0:
            save_checkpoint(Path(out) / "recovery.pt", model, optimizer, state, meta)
        group = order[group_start : group_start + c["accumulation"]]
        optimizer.zero_grad(set_to_none=True)
        records = []
        try:
            with attention_context(c["attention_backend"]):
                for position, i in enumerate(group, group_start):
                    # Independent sample mask generator leaves dropout RNG unaffected.
                    mg = torch.Generator().manual_seed(c["seed"] + epoch * 1000003 + i)
                    mask = choose_mask(len(runs[i]["ids"]), c["mask_fraction"], mg)
                    record = dict(
                        epoch=epoch,
                        update=state["update"] + 1,
                        position=position,
                        run_id=runs[i]["run_id"],
                        index=i,
                        masked_positions=mask.tolist(),
                    )
                    records.append(record)
                    ids, a, target_ids, target = prepare_sample(runs[i], mask, device)
                    pred = model(ids, a)
                    check_tensor(pred, "training predictions")
                    loss = masked_loss(pred, target_ids, target)
                    check_tensor(loss, "training loss")
                    (loss / len(group)).backward()
                    stats = gradient_summary(model)
                    if stats["max_abs"] > c["gradient_warning"]:
                        raise DiagnosticStop(
                            "Large finite gradient: diagnostic threshold exceeded; no clipping applied"
                        )
                    state["total_loss"] += float(loss.detach())
                    state["count"] += 1
            stats = gradient_summary(model)
            lr = learning_rate(state["update"] + 1, c)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            check_state(model, optimizer)
            state["update"] += 1
            state["position"] = group_start + len(group)
            if state["update"] % 63 == 0 or state["position"] == len(order):
                print(
                    f"Epoch {epoch:03d}: {state['position']:,}/{len(order):,}; MSE={state['total_loss']/state['count']:.7g}; lr={lr:.7g}; grad_norm={stats['l2']:.7g}",
                    flush=True,
                )
                monitor.event(
                    dict(
                        epoch=epoch,
                        update=state["update"],
                        position=state["position"],
                        **stats,
                    )
                )
        except (DiagnosticStop, RuntimeError) as exc:
            atomic_save(
                dict(
                    reason=str(exc),
                    records=records,
                    recent=list(monitor.recent),
                    recovery="recovery.pt",
                    note="Recovery is pre-update; failure bundle does not pretend post-failure weights are pre-failure weights.",
                ),
                Path(out) / "failure.pt",
            )
            raise
    return state["total_loss"] / state["count"]
