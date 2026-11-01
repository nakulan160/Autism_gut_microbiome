"""Compact output history and standalone dashboard; no external web dependencies."""

import base64
import io
import json
import os
import zlib
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def read_history(path):
    if not Path(path).exists():
        return {"metrics": [], "predictions": {}}
    return json.loads(zlib.decompress(Path(path).read_bytes()))


def update_history(out, epoch, train_mse, validation_mse, pairs=None):
    path = Path(out) / "history.json.zlib"
    history = read_history(path)
    history["metrics"] = [r for r in history["metrics"] if r["epoch"] < epoch]
    history["predictions"] = {
        k: v for k, v in history["predictions"].items() if int(k) < epoch
    }
    history["metrics"].append(
        dict(epoch=epoch, train_mse=train_mse, validation_mse=validation_mse)
    )
    if pairs is not None:
        history["predictions"][str(epoch)] = pairs
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(
        zlib.compress(json.dumps(history, allow_nan=False).encode(), level=6)
    )
    os.replace(tmp, path)
    return history


def plot_results(out, epoch, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(out)
    rows = history["metrics"]
    fig, ax = plt.subplots()
    ax.plot(
        [r["epoch"] for r in rows], [r["train_mse"] for r in rows], label="Training"
    )
    ax.plot(
        [r["epoch"] for r in rows],
        [r["validation_mse"] for r in rows],
        label="Validation",
    )
    ax.set(xlabel="Epoch", ylabel="Mean sample MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    stream = io.BytesIO()
    fig.savefig(stream, format="png")
    plt.close(fig)
    loss_image = base64.b64encode(stream.getvalue()).decode()
    panels = []
    for key, pairs in sorted(
        history["predictions"].items(), key=lambda item: int(item[0])
    ):
        a = np.asarray(pairs)
        actual, pred = a[:, 0], a[:, 1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(actual, pred, s=3, alpha=0.3)
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        axes[0].plot([lo, hi], [lo, hi], "k--")
        axes[0].set(xlabel="Original masked abundance", ylabel="Predicted abundance")
        axes[1].hist(pred - actual, bins=50)
        axes[1].set(xlabel="Prediction minus original abundance", ylabel="Count")
        fig.suptitle("Validation predictions: epoch " + key)
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        plt.close(fig)
        panels.append((key, base64.b64encode(stream.getvalue()).decode()))
    options = "".join(
        '<option value="' + k + '">Epoch ' + k + "</option>" for k, _ in panels
    )
    images = "".join(
        '<img class="panel" id="epoch-'
        + k
        + '" alt="Prediction and residual plots epoch '
        + k
        + '" src="data:image/png;base64,'
        + image
        + '" style="display:none;max-width:100%">'
        for k, image in panels
    )
    table = "".join(
        "<tr><td>"
        + str(r["epoch"])
        + "</td><td>"
        + format(r["train_mse"], ".8g")
        + "</td><td>"
        + format(r["validation_mse"], ".8g")
        + "</td></tr>"
        for r in rows
    )
    html = """<!doctype html><meta charset="utf-8"><title>Microbiome training results</title>
<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:6px 20px;text-align:right}select{font-size:18px}img{max-width:100%}</style>
<h1>Training and validation results</h1><p>Updated through epoch EPOCH. This dashboard works offline.</p>
<h2>Loss history</h2><img alt="Training and validation loss" src="data:image/png;base64,LOSS">
<h2>Prediction and residual history</h2><label>Choose saved epoch: <select id="epoch">OPTIONS</select></label>IMAGES
<h2>How to read these plots</h2><ul><li>Lower validation MSE means smaller mean squared errors on the fixed masked validation task.</li><li>Training improving while validation worsens can indicate overfitting; inspect the full trend.</li><li>Scatter points close to the dashed diagonal indicate accurate abundance predictions.</li><li>Residual is prediction minus target. Positive values mean overprediction; negative values mean underprediction.</li><li>Prediction plots use at most 10,000 validation masked pairs in fixed order. All validation samples contribute to loss.</li><li>Run-level random validation does not establish donor-independent or cross-study generalization.</li></ul>
<h2>Exact epoch metrics</h2><table><tr><th>Epoch</th><th>Training MSE</th><th>Validation MSE</th></tr>TABLE</table>
<script>const s=document.getElementById('epoch');function show(){document.querySelectorAll('.panel').forEach(x=>x.style.display='none');const p=document.getElementById('epoch-'+s.value);if(p)p.style.display='block'}s.onchange=show;if(s.options.length)s.selectedIndex=s.options.length-1;show();</script>
"""
    html = (
        html.replace("EPOCH", str(epoch))
        .replace("LOSS", loss_image)
        .replace("OPTIONS", options)
        .replace("IMAGES", images)
        .replace("TABLE", table)
    )
    tmp = out / "dashboard.tmp"
    tmp.write_text(html)
    os.replace(tmp, out / "dashboard.html")
