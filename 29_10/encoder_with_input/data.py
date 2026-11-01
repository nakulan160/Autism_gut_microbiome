import csv
import gzip
import hashlib
import math
import torch


def load_data(path, excluded, dtype, abundance_max, log_every):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1048576), b''):
            digest.update(chunk)
    opener = gzip.open if str(path).endswith('.gz') else open
    runs, seen = [], set()
    with opener(path, 'rt', newline='') as file:
        rows = csv.reader(file, delimiter='\t')
        header = next(rows)
        if len(set(header)) != len(header):
            raise ValueError('Duplicate columns')
        columns = [j for j in range(1, len(header)) if header[j] not in excluded]
        vocabulary = [header[j] for j in columns]
        if not vocabulary:
            raise ValueError('Empty vocabulary')
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f'Wrong column count: {row[:1]}')
            run_id = row[0]
            if not run_id or run_id in seen:
                raise ValueError(f'Duplicate/empty run: {run_id}')
            seen.add(run_id)
            # Skip blanks, but retain reported zeros. IDs start at 1.
            pairs = [(i, float(row[j])) for i, j in enumerate(columns, 1)
                     if row[j].strip() != '']
            if not pairs or any(not math.isfinite(a) or not 0 <= a <= abundance_max for _, a in pairs):
                raise ValueError(f'{run_id}: empty sample or invalid abundance')
            ids, values = zip(*pairs)
            runs.append(dict(run_id=run_id, ids=torch.tensor(ids, dtype=torch.long),
                             values=torch.tensor(values, dtype=dtype)))
            if len(runs) % log_every == 0:
                print(f'Loaded {len(runs):,} runs', flush=True)
    if len(runs) < 2:
        raise ValueError('Need at least two runs')
    return runs, vocabulary, digest.hexdigest()


def make_split(n, fraction, seed):
    # Return training indices first, validation indices second.
    nv = min(n - 1, max(1, math.ceil(n * fraction)))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    return order[nv:], order[:nv]


def choose_mask(n, fraction, generator):
    count = max(1, math.floor(n * fraction))
    return torch.randperm(n, generator=generator)[:count]


def prepare_sample(run, masked, device):
    # Return visible inputs, followed by masked IDs and original targets.
    visible = torch.ones(len(run['ids']), dtype=torch.bool)
    visible[masked] = False
    return (run['ids'][visible].to(device), run['values'][visible].to(device),
            run['ids'][masked].to(device), run['values'][masked].to(device))
