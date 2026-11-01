import csv, gzip, hashlib, math
import torch


def load_data(path, excluded):
    """TSV first column=run ID; remaining columns=species. Blank=absent; reported zero is retained."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            digest.update(b)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError("Duplicate columns")
        columns = [
            (j, name) for j, name in enumerate(header[1:], 1) if name not in excluded
        ]
        vocabulary = [name for _, name in columns]
        if not vocabulary:
            raise ValueError("Empty vocabulary")
        runs = []
        seen = set()
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Wrong column count: {row[:1]}")
            run = row[0]
            if not run or run in seen:
                raise ValueError(f"Duplicate/empty run: {run}")
            seen.add(run)
            ids = []
            values = []
            for species_id, (j, name) in enumerate(columns, 1):
                value = row[j].strip()
                if value == "":
                    continue
                a = float(value)
                if not math.isfinite(a) or not 0 <= a <= 100:
                    raise ValueError(
                        f"{run}/{name}: abundance outside [0,100]: {value}"
                    )
                ids.append(species_id)
                values.append(a)
            if not ids:
                raise ValueError(
                    f"{run}: no reported species after exclusions; explicit decision required"
                )
            runs.append(
                dict(
                    run_id=run,
                    ids=torch.tensor(ids, dtype=torch.long),
                    values=torch.tensor(values, dtype=torch.float32),
                )
            )
            if len(runs) % 5000 == 0:
                print(f"Loaded {len(runs):,} runs", flush=True)
    if len(runs) < 2:
        raise ValueError("Need at least two runs")
    return runs, vocabulary, digest.hexdigest()
