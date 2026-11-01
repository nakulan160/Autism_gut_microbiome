# GMrepo reported-abundance Transformer

This package removes the exact `Unknown` feature and trains a randomly initialized,
abundance-only Transformer to reconstruct randomly hidden reported abundances.
All 36,382 supplied runs are training samples. There is no validation or test split.

## Data after removing Unknown

| Item | Value |
|---|---:|
| Runs retained | 36,382 |
| Original feature columns | 3,424 |
| Removed feature columns | 1 (`Unknown`) |
| Remaining feature columns / vocabulary size | 3,423 |
| Total columns including `run_id` | 3,424 |
| Reported cells removed with `Unknown` | 35,015 |
| Reported cells retained | 3,142,645 |
| Smallest / largest retained profile | 1 / 300 reported entries |
| Runs with just one retained entry | 30 |

The filtered file is `data/non_asd_qc_pass_reported_no_unknown.tsv.gz`.
Every remaining abundance string and blank cell is preserved. No zero filling,
log transformation, normalization, abundance renormalization, name merging, or
additional taxonomy filtering is performed. Removing Unknown does not rescale the
other abundances. Other reported labels, including genus-level labels, remain.
Thus, 3,423 is the number of retained reported features, not a claim that every
column is a taxonomically verified species.

Training parses reported numeric strings as float32. The exported data retains the
original strings; normal floating-point representation limits apply in the model.
The original input file is not modified.

## Run

Python 3.10 or newer is required. Install the dependencies in the environment you
will use for training:

```bash
python -m pip install -r requirements.txt
```

For a particular CUDA version, install the appropriate PyTorch build using its
[official installation selector](https://pytorch.org/get-started/locally/), then
install NumPy and Matplotlib in that same environment.

From the extracted package folder, train using the included filtered data:

```bash
python train_masked_abundance.py --config settings.json --input data/non_asd_qc_pass_reported_no_unknown.tsv.gz --output-dir results
```

Alternatively, from your original `15_10` folder, place the Python file there and
use your existing export. The code removes Unknown automatically:

```bash
python train_masked_abundance.py --input metagenomic_totla/non_asd_qc_pass_reported_blank.tsv.gz --output-dir metagenomic_totla/transformer_no_unknown
```

`--device auto` selects CUDA when available, otherwise CPU. You may explicitly use
`--device cpu` or `--device cuda`. `--threads 10` controls CPU threads.

To only create the filtered data, vocabulary, and audit without training:

```bash
python train_masked_abundance.py --input metagenomic_totla/non_asd_qc_pass_reported_blank.tsv.gz --output-dir metagenomic_totla/transformer_no_unknown --prepare-only
```

Then use the same command without `--prepare-only` to train.

## Exact species/output matching

`species_vocabulary.tsv` records every retained original label in column order:

- CLS has conceptual ID 0 and is input-only.
- Feature IDs are 1 through 3,423.
- The output contains 3,423 numbers, indexed from 0 through 3,422.
- Feature ID `s` always selects output index `s - 1`.

IDs stay outside the encoder. A masked ID list and its original abundance targets
are saved before removing those entries from the input. Predictions are gathered
using that same ID list. Unreported cells and unmasked entries do not enter the
loss. Checkpoints contain the complete ordered vocabulary and its hash; restoring
with a different mapping or different retained profiles is rejected.

## Masking and loss

For a profile with `n` reported entries, select
`k = max(1, floor(0.10 * n))` entries randomly without replacement. New masks are
drawn on each training visit. For fewer than ten entries this selects one entry.

The code saves selected IDs and original targets, constructs a binary keep mask,
multiplies abundances by that mask, and removes entries using the mask. It never
removes an entry merely because its numerical abundance equals zero.

For the 30 single-entry runs, masking the sole entry leaves only CLS as model
input and one original abundance as the target. There is no added input value.
Those runs contribute learning about a prior rather than context-dependent
reconstruction. No sample is dropped. An empty retained profile in a future file
causes an explicit error because it has no reconstruction target.

For each sample, loss is the sum of squared errors on its `k` masked targets,
divided by `k`. Four masked entries means divide by four. For an optimizer update,
the sample losses of 16 individually processed runs are averaged. The final partial
group uses its actual number of runs. There is no sequence padding.

Epoch MSE is the sum of per-sample masked MSEs divided by the number of samples.
Samples have equal weight regardless of the number of masked targets. Epoch RMSE
is the square root of that epoch MSE. Epoch MAE also averages the per-sample means.
Training metrics are collected during training, while weights change and encoder
dropout is active. Diagnostic plots use evaluation mode at the plotting epoch.

## Model and optimizer

Each scalar abundance is linearly projected from 1 to 512 dimensions. The projected
embedding is scaled by sqrt(512), following the agreed embedding scaling; raw data
are not normalized. One shared learned CLS vector is initialized once, placed
first, and trained with the other parameters. Its representation is extracted
after the sixth encoder block.

The encoder uses six independently initialized PyTorch `TransformerEncoderLayer`
blocks with 512 dimensions, eight heads, a 2,048-dimensional feed-forward hidden
layer, ReLU, dropout 0.1, residual connections, and normalization after each
residual addition (`norm_first=False`). Each head has dimension 64. No positional
encoding, species identity embedding, input dropout, causal mask, or padding is
used. Internal encoder dropout remains enabled during training.

The CLS prediction head is:

1. Linear 512 to 1,024, then ReLU.
2. Linear 1,024 to 2,048, then ReLU.
3. Linear 2,048 to vocabulary size, with a linear output (option A).

Output predictions may be negative. Losses, diagnostic tables, and plots use raw
predictions with no clipping or output renormalization.

Adam uses beta1 0.9, beta2 0.98, epsilon 1e-9, zero weight decay, and no gradient
clipping. The learning rate follows the original Transformer warmup/inverse-square-root
formula, with 4,000 optimizer updates of warmup. One optimizer update happens after
accumulating up to 16 sample gradients. With 36,382 runs, there are 2,274 updates per
epoch. The update count, not the sample count or epoch number, drives the schedule.

The original paper describes an encoder-decoder translation model. CLS, continuous
abundance projection, the reconstruction head, and the requested omissions are
adaptations. This package follows the agreed base encoder settings rather than
claiming to reproduce the complete translation model.

Because inputs contain only abundance values and no species or position signal,
the encoder cannot distinguish two profiles that differ only in which species
owns each visible value. In evaluation mode, reordering visible values gives the
same CLS representation up to floating-point rounding. Exact output/target identity
matching still holds, but that bookkeeping does not provide species identities to
the encoder. Interpret representations with this design limitation in mind.

## Outputs and schedules

Defaults are 100 epochs, a checkpoint every five completed epochs, and plots every
ten completed epochs. These intervals are independent.

| Output | When / contents |
|---|---|
| `epoch_metrics.csv` | Every completed epoch: MSE, RMSE, MAE, counts, learning rate, duration |
| `masked_targets/epoch_0001.tsv.gz`, etc. | Every epoch: run ID, masked feature ID, output index, original abundance string, prediction |
| `checkpoints/epoch_0005.pt`, etc. | Epochs 5, 10, 15, ..., 100 |
| `plots/epoch_0010_loss.png`, etc. | Every ten epochs; loss curve uses every stored epoch through that point |
| `plots/epoch_0010_predictions.png`, etc. | Original vs predicted masked abundances, with identity line |
| `plots/epoch_0010_residuals.png`, etc. | Predicted abundance vs residual; residual = prediction minus original |
| `diagnostics/epoch_0010.tsv.gz`, etc. | Exact species-matched values behind the corresponding plots |
| `diagnostic_masks.json` | Fixed training sample IDs and masks used for comparable diagnostic snapshots |
| `vocabulary.json`, `species_vocabulary.tsv` | Full feature/output mapping |
| `input_audit.json`, `settings_used.json` | Input checks, data fingerprint, effective settings |

The fixed diagnostic subset defaults to 128 training runs. They continue to train
normally. Diagnostic masks use a separate random generator so plotting does not
alter training masks. These are training reconstruction plots and do not measure
performance on unseen samples.

All scheduled checkpoints are retained. For the default architecture a full
model-plus-Adam checkpoint is approximately 340 MB, so twenty checkpoints require
approximately 7 GB, in addition to logs and data. Exact sizes depend on settings.

## Resume

Use the same input, settings, and output folder:

```bash
python train_masked_abundance.py --config settings.json --input data/non_asd_qc_pass_reported_no_unknown.tsv.gz --output-dir results --resume latest
```

A checkpoint contains model and optimizer states, learning-rate update number,
history, vocabulary, data fingerprint, diagnostic masks, and Python/NumPy/PyTorch
random states. Resume restores these and starts with the next epoch. Metrics/logs
from uncheckpointed epochs are replayed; their per-epoch files are archived under
`interrupted_outputs` before replay. Only load a checkpoint you trust.

You may increase `epochs` when resuming. Changing architecture, masking, accumulation,
optimizer, or diagnostic sampling settings is rejected to prevent accidental
continuation of a different experiment. Continue with `latest`, or use a fresh output
folder when deliberately restoring an older checkpoint. Exact numerical reproducibility
can depend on device, PyTorch version, and platform; CPU resume equivalence was tested.

## Customize one part

Settings priority is: `Config` defaults, then JSON settings, then explicit CLI
arguments. Edit `settings.json` for numerical settings when using `--config`.
For algorithm changes, edit the corresponding isolated function or model class:

| Main function | Responsibility |
|---|---|
| `build_vocabulary` | Feature exclusions and permanent ID assignment |
| `load_profiles` | Reported-only loading and exact filtered export |
| `mask_profile` | Random selection, target saving, binary masking, removal |
| `prepare_batch` | One variable-length sample converted to tensors |
| `build_model` | Construct the encoder and head |
| `compute_masked_loss` | Exact ID matching and per-sample mean squared error |
| `train_one_epoch` | Accumulation, updates, training metrics, mask logs |
| `collect_training_diagnostics` | Fixed training reconstruction snapshots |
| `save_checkpoint` | Save complete continuation state |
| `load_checkpoint` | Validate and restore continuation state |
| `save_results` | Persist epoch metrics and diagnostic values |
| `plot_training_loss` | Epoch vs loss |
| `plot_predictions` | Original vs predicted abundances |
| `plot_residuals` | Residual plot |
| `run_training` | Complete workflow and schedules |

Model classes are `AbundanceEmbedding`, `CLSEncoder`, and `PredictionHead`.
Small helpers handle file I/O, configuration, learning-rate calculation, and RNGs.

## Verification

```bash
python -m unittest -v test_training.py
```

Tests cover unchanged retained cells, blanks and original zeros, species/output
matching with unsorted IDs, averaging four masked targets, single-entry CLS-only
inputs, gradient accumulation including a partial group, permutation invariance,
checkpoint identity checks, exact CPU resume equivalence, and plot scheduling.
`verification_report.json` records the checks completed for the supplied data/code.
The full 100-epoch training experiment has not been run in this package preparation.

References: [original Transformer paper](https://arxiv.org/abs/1706.03762),
[PyTorch encoder layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html),
[Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html), and
[saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
