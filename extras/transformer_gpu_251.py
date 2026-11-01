from pathlib import Path
import argparse
import gc
import json
import math
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "10"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

# Keep both requested code files together: this imports the shared checks/metrics.
# It does not import CUDA ML libraries or run ML training.
from ml_gpu_251 import ROOT, SEED, METRICS, DATA_HASHES, load_data, make_folds, calculate_metrics, save_summary, versions

EMBED_DIM = 512
LAYERS = 6
HEADS = 8
FF_DIM = 2048
DROPOUT = 0.2
LOG_EDGES = np.logspace(-12, 0, 1001, dtype=np.float64)


def abundance_ids(percent):
    # Fixed bins are defined before CV. No test-fold statistics are fitted.
    fraction = np.asarray(percent, dtype=np.float64) / 100.0
    if not np.isfinite(fraction).all() or (fraction < 0).any() or (fraction > 1).any():
        raise ValueError("Expected finite percentage abundances in [0,100]")
    if ((fraction > 0) & (fraction < LOG_EDGES[0])).any():
        raise ValueError("A positive abundance falls below the agreed 1e-12 lower bin limit")
    result = np.searchsorted(LOG_EDGES[1:-1], fraction, side="right") + 1
    result[fraction == 0] = 0
    return result.astype(np.int64)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FullBatchSampler:
    # BatchNorm cannot train on a singleton. Merge a final size-1 batch into
    # the preceding batch (size 17 at the default 16), keeping every sample.
    def __init__(self, count, batch_size, seed):
        if count < 2 or batch_size < 2:
            raise ValueError("At least two training samples and batch size >=2 are required")
        self.count, self.batch_size = count, batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(self.count, generator=self.generator).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.count, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        total = math.ceil(self.count / self.batch_size)
        return total - int(self.count % self.batch_size == 1 and total > 1)


class SpeciesTransformer(nn.Module):
    def __init__(self, variant, checkpoint_layers=False):
        super().__init__()
        if variant not in ["addition", "concatenation"]:
            raise ValueError("Unknown embedding variant")
        self.variant = variant
        self.d_model = 512 if variant == "addition" else 1024
        self.checkpoint_layers = checkpoint_layers
        # Species dictionary: 0=CLS, 1..827=frozen species order.
        self.species_embedding = nn.Embedding(828, 512)
        # Abundance dictionary: 0=zero; 1..1000=positive log bins.
        self.abundance_embedding = nn.Embedding(1001, 512, padding_idx=0)
        self.register_buffer("species_ids", torch.arange(828, dtype=torch.long))
        positions = torch.arange(828, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / self.d_model))
        positional = torch.zeros(828, self.d_model)
        positional[:, 0::2] = torch.sin(positions * frequency)
        positional[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer("positional_encoding", positional.unsqueeze(0))
        self.input_dropout = nn.Dropout(DROPOUT)
        # Independent layers: self-attention -> residual/LayerNorm -> FFN ->
        # residual/LayerNorm. No decoder, cross-attention or causal mask.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=HEADS,
                                       dim_feedforward=FF_DIM, dropout=DROPOUT,
                                       activation="relu", batch_first=True, norm_first=False)
            for _ in range(LAYERS)
        ])
        hidden1, hidden2 = self.d_model // 2, self.d_model // 4
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden2, 2),
        )
        # Explicit implementation initialization; not a paper-derived new result.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.abundance_embedding.weight[0].zero_()

    def fused_tokens(self, values):
        if values.ndim != 2 or values.shape[1] != 827 or values.dtype != torch.long:
            raise ValueError("Expected integer abundance IDs with shape (batch,827)")
        batch = len(values)
        species = self.species_embedding(self.species_ids).unsqueeze(0).expand(batch, -1, -1)
        cls_zero = torch.zeros(batch, 1, dtype=torch.long, device=values.device)
        abundance = self.abundance_embedding(torch.cat([cls_zero, values], dim=1))
        if self.variant == "addition":
            return species + abundance
        return torch.cat([species, abundance], dim=-1)

    def forward(self, values):
        tokens = self.fused_tokens(values)
        # The paper scales input embeddings and adds sinusoidal positions.
        # This scales embeddings, not raw abundance times a species vector.
        hidden = self.input_dropout(tokens * math.sqrt(self.d_model) + self.positional_encoding)
        for layer in self.layers:
            if self.checkpoint_layers and self.training:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        sample = hidden[:, 0, :]
        return self.classifier(sample)


def write_dictionaries(vocabulary, output):
    species = pd.DataFrame({"Token_ID": np.arange(1, 828),
                            "Current_NCBI_TaxID": vocabulary["Current_NCBI_TaxID"],
                            "Species_name": vocabulary["Current_Species_Name"],
                            "Feature_header": vocabulary["Feature_Header"]})
    cls = pd.DataFrame([{"Token_ID": 0, "Current_NCBI_TaxID": "", "Species_name": "[CLS]", "Feature_header": "[CLS]"}])
    pd.concat([cls, species], ignore_index=True).to_csv(output / "species_dictionary.tsv", sep="\t", index=False)
    mapping = {"0": {"name": "[CLS]", "taxid": None}}
    mapping.update({str(i + 1): {"name": str(r.Current_Species_Name), "taxid": int(r.Current_NCBI_TaxID)}
                    for i, r in vocabulary.iterrows()})
    (output / "species_dictionary.json").write_text(json.dumps(mapping, indent=2) + "\n")
    bins = pd.DataFrame({"Abundance_ID": np.arange(1, 1001), "Lower_fraction": LOG_EDGES[:-1],
                         "Upper_fraction": LOG_EDGES[1:]})
    bins["Lower_percent"] = bins.Lower_fraction * 100
    bins["Upper_percent"] = bins.Upper_fraction * 100
    bins["Upper_inclusive"] = bins.Abundance_ID.eq(1000)
    zero = pd.DataFrame([{"Abundance_ID": 0, "Lower_fraction": 0.0, "Upper_fraction": 0.0,
                         "Lower_percent": 0.0, "Upper_percent": 0.0, "Upper_inclusive": True}])
    pd.concat([zero, bins], ignore_index=True).to_csv(output / "abundance_dictionary.tsv", sep="\t", index=False)


def self_test(device="cpu"):
    # Synthetic software check, not model evaluation on ASD data.
    torch.set_num_threads(10)
    sample = np.zeros((2, 827))
    sample[0, :4] = [0.00002, 0.01323, 1.0, 100.0]
    sample[1, :4] = [0.00003, 0.01, 2.0, 50.0]
    ids = torch.from_numpy(abundance_ids(sample)).to(device)
    for variant in ["addition", "concatenation"]:
        set_seed(42)
        model = SpeciesTransformer(variant, checkpoint_layers=True).to(device)
        tokens = model.fused_tokens(ids)
        assert tuple(tokens.shape) == (2, 828, model.d_model)
        if variant == "concatenation":
            assert torch.count_nonzero(tokens[:, 0, 512:]).item() == 0
        assert torch.count_nonzero(model.abundance_embedding.weight[0]).item() == 0
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        logits = model(ids)
        assert tuple(logits.shape) == (2, 2)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1], device=device))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.species_embedding.weight.grad.abs().sum().item() > 0
        assert model.abundance_embedding.weight.grad[1:].abs().sum().item() > 0
        assert torch.count_nonzero(model.abundance_embedding.weight.grad[0]).item() == 0
        optimizer.step()
        assert torch.count_nonzero(model.abundance_embedding.weight[0]).item() == 0
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(ids)).all()
        print("Software check passed:", variant, "width", model.d_model,
              "parameters", sum(p.numel() for p in model.parameters()), flush=True)
        del optimizer, model, tokens, logits, loss
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
    for count in [225, 226]:
        batches = list(FullBatchSampler(count, 16, 42))
        assert sorted(i for b in batches for i in b) == list(range(count))
        assert all(len(b) > 1 for b in batches)
    print("Sampler preserves every training sample; zero abundance embedding remains fixed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GPU 10-fold addition/concatenation species Transformer")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", choices=["addition", "concatenation", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", default="transformer_gpu")
    parser.add_argument("--amp", action="store_true", help="Explicitly enable CUDA float16 mixed precision")
    parser.add_argument("--checkpoint-layers", action="store_true", help="Reduce activation memory without changing layer dimensions")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save final per-fold weights; requires several GB")
    parser.add_argument("--check-only", action="store_true", help="Verify data, fold manifest and GPU; do not train")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic full-architecture CPU checks only")
    args = parser.parse_args()
    torch.set_num_threads(10)
    if args.self_test:
        self_test()
        return
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be positive and batch-size >=2")
    x, y, metadata, vocabulary = load_data(args.root)
    folds = make_folds(x, y, metadata, args.root)
    ids = torch.from_numpy(abundance_ids(x))
    if not torch.cuda.is_available() or args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError("CUDA PyTorch and an available NVIDIA GPU are required; no CPU training fallback is used")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print("GPU:", torch.cuda.get_device_name(args.gpu), "| torch", torch.__version__,
          "| CUDA build", torch.version.cuda, flush=True)
    if args.check_only:
        print("Verified 251×827 inputs, both embedding dimensions, fixed bins and shared ten folds.")
        return
    output = args.root / "outputs" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    write_dictionaries(vocabulary, output)
    selected = ["addition", "concatenation"] if args.variant == "both" else [args.variant]
    config = {"input_sha256": DATA_HASHES, "versions": versions(), "torch_cuda": torch.version.cuda,
              "GPU_name": torch.cuda.get_device_name(args.gpu), "GPU": args.gpu,
              "seed": 42, "fold_seed": "42+one_based_fold", "layers": LAYERS,
              "attention_heads": HEADS, "feedforward": FF_DIM, "activation": "ReLU",
              "normalization": "post-sublayer LayerNorm; BatchNorm in two classifier hidden layers",
              "dropout": DROPOUT, "encoder_only": True, "causal_mask": False,
              "zero_abundance_attention_mask": False, "species_embedding": "nn.Embedding(828,512)",
              "abundance_embedding": "nn.Embedding(1001,512,padding_idx=0)",
              "abundance_binning": "percent/100; zero ID0; 1000 fixed logarithmic bins from1e-12to1",
              "position_encoding": "sin/cos on frozen species order; CLS at position0",
              "input_embedding_scale": "sqrt(d_model)", "initialization": "Xavier-uniform matrix weights; PyTorch default vector parameters",
              "epochs": args.epochs, "batch_size": args.batch_size,
              "singleton_batch_rule": "merge final singleton into prior batch; no samples dropped",
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "loss": "CrossEntropyLoss without label smoothing or class weights",
              "gradient_clip_norm": 1.0, "early_stopping": False,
              "AMP": args.amp, "activation_checkpointing": args.checkpoint_layers,
              "SD": "sample SD across10folds,ddof1", "project_holdouts": False,
              "variants": {}}
    all_rows, all_predictions = [], []
    for variant in selected:
        variant_output = output / variant
        variant_output.mkdir()
        rows, predictions, history = [], [], []
        for fold, (train, test) in enumerate(folds, 1):
            seed = SEED + fold
            set_seed(seed)
            model = SpeciesTransformer(variant, args.checkpoint_layers).to(device)
            config["variants"][variant] = {"d_model": model.d_model,
                                          "head_dimension": model.d_model // HEADS,
                                          "classifier": [model.d_model, model.d_model // 2, model.d_model // 4, 2],
                                          "parameters": sum(p.numel() for p in model.parameters())}
            (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
            train_data = TensorDataset(ids[train], torch.from_numpy(y[train]))
            loader = DataLoader(train_data, batch_sampler=FullBatchSampler(len(train), args.batch_size, seed),
                                num_workers=0, pin_memory=True)
            test_loader = DataLoader(TensorDataset(ids[test]), batch_size=args.batch_size,
                                     num_workers=0, pin_memory=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss, seen = 0.0, 0
                for batch, labels in loader:
                    batch = batch.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                        logits = model(batch)
                        loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * len(batch)
                    seen += len(batch)
                if seen != len(train):
                    raise ValueError("Not every training sample was used")
                history.append({"Variant": variant, "Fold": fold, "Epoch": epoch,
                                "Train_loss": total_loss / seen, "Train_N": seen})
                pd.DataFrame(history).to_csv(variant_output / "training_history.tsv", sep="\t", index=False)
                print(variant, f"fold{fold}/10 epoch{epoch}/{args.epochs}",
                      f"train_loss={total_loss / seen:.5f}", flush=True)
            # The held-out fold is evaluated only after the fixed training duration.
            model.eval()
            scores = []
            with torch.no_grad():
                for (batch,) in test_loader:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                        logits = model(batch.to(device, non_blocking=True))
                    scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            score = np.concatenate(scores)
            predicted = (score > 0.5).astype(int)
            result = calculate_metrics(y[test], predicted, score)
            record = {"Model": variant, "Fold": fold, "Train_N": len(train), "Test_N": len(test),
                      "Seconds": time.perf_counter() - start, **result}
            rows.append(record)
            predictions.extend({"Model": variant, "Run ID": metadata.index[i], "Fold": fold,
                                "Label": int(y[i]), "Predicted": int(predicted[j]), "ASD_probability": float(score[j])}
                               for j, i in enumerate(test))
            pd.DataFrame(rows).to_csv(variant_output / "fold_metrics.tsv", sep="\t", index=False)
            pd.DataFrame(predictions).to_csv(variant_output / "oof_predictions.tsv", sep="\t", index=False)
            print(variant, f"fold{fold}/10 TEST:", ", ".join(f"{m}={result[m]:.4f}" for m in METRICS), flush=True)
            if args.save_checkpoints:
                torch.save({"state_dict": model.state_dict(), "config": config, "fold": fold,
                            "variant": variant}, variant_output / f"fold_{fold:02d}_final.pt")
            if torch.count_nonzero(model.abundance_embedding.weight[0]).item() != 0:
                raise ValueError("Zero abundance embedding changed")
            del model, optimizer, loader, test_loader, scaler, train_data
            gc.collect()
            torch.cuda.empty_cache()
        save_summary(rows, variant_output)
        table = pd.DataFrame(predictions)
        if len(table) != 251 or not table["Run ID"].is_unique:
            raise ValueError("Incomplete out-of-fold predictions")
        pd.DataFrame([{"Model": variant, **calculate_metrics(table.Label.to_numpy(), table.Predicted.to_numpy(), table.ASD_probability.to_numpy())}]).to_csv(
            variant_output / "pooled_oof_metrics.tsv", sep="\t", index=False)
        all_rows.extend(rows)
        all_predictions.extend(predictions)
    save_summary(all_rows, output, filename="variants_mean_sd.tsv")
    pd.DataFrame(all_rows).to_csv(output / "all_fold_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(all_predictions).to_csv(output / "all_oof_predictions.tsv", sep="\t", index=False)
    (output / "COMPLETED").write_text("Requested encoder variants completed ten folds each.\n")


if __name__ == "__main__":
    main()
