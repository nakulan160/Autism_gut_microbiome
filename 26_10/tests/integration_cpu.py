import json, pathlib, subprocess, sys, tempfile, torch

base = pathlib.Path(__file__).resolve().parents[1]
tmp = pathlib.Path(tempfile.mkdtemp(prefix="microbiome_integration_"))
data = tmp / "input.tsv"
data.write_text(
    "run_id\tA\tB\tC\tUnknown\n"
    + "".join(f"R{i}\t{0.03 if i%2 else 0}\t{2+i}\t{88-i}\t1\n" for i in range(10))
)
c = json.loads((base / "settings.json").read_text())
c.update(
    input=str(data),
    output=str(tmp / "results"),
    epochs=2,
    species_dim=8,
    abundance_dim=8,
    layers=1,
    heads=2,
    feedforward=32,
    head_dims=[16],
    accumulation=3,
    checkpoint_every=1,
    plot_every=1,
    recovery_every_updates=2,
)
config = tmp / "config.json"
config.write_text(json.dumps(c))


def run(args, success=True):
    r = subprocess.run(
        [sys.executable, *args], cwd=base, capture_output=True, text=True
    )
    if (r.returncode == 0) != success:
        raise RuntimeError(r.stdout + r.stderr)
    print(r.stdout[-700:])
    return r


run(["main.py", "--config", str(config), "--device", "cpu"])
assert len(list((tmp / "results").rglob("*"))) > 0
assert not (tmp / "results/metrics.csv").exists()
assert not (tmp / "results/split.json").exists()
import zlib

h = json.loads(zlib.decompress((tmp / "results/history.json.zlib").read_bytes()))
assert len(h["metrics"]) == 2 and set(h["predictions"]) == {"1", "2"}
assert "epoch-1" in (tmp / "results/dashboard.html").read_text()
assert (tmp / "results/loss.png").exists()
assert len([x for x in (tmp / "results").rglob("*") if x.is_file()]) == 8
final = torch.load(tmp / "results/checkpoints/epoch_0002.pt", weights_only=False)
run(
    [
        "main.py",
        "--config",
        str(config),
        "--device",
        "cpu",
        "--resume",
        str(tmp / "results/checkpoints/epoch_0001.pt"),
    ]
)
again = torch.load(tmp / "results/checkpoints/epoch_0002.pt", weights_only=False)
assert all(
    torch.equal(v, again["model"][k]) for k, v in final["model"].items()
), "Resume mismatch"
# Force a finite-gradient diagnostic stop and exercise replay.
c["output"] = str(tmp / "failure")
c["gradient_warning"] = 1e-20
config.write_text(json.dumps(c))
run(["main.py", "--config", str(config), "--device", "cpu"], False)
run(["replay.py", "--results", c["output"], "--device", "cpu"])
assert (tmp / "failure/replay_trace.jsonl").stat().st_size > 0
print(
    "PASS: full training, plots, checkpoint resume bitwise equality, finite-gradient failure capture, operation replay"
)
print(tmp)
