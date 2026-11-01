# Modular masked-abundance encoder

## Agreed experiment

Species name -> stable 1-based ID -> nn.Embedding(256).
Reported abundance in percent [0,100] -> nearest integer, halfway upward -> nn.Embedding(101,256).
Concatenate to 512. Bin 0 is a normal trainable category, never padding. No unreported species are added.
The input abundance is NOT multiplied by embedding weights. No extra sqrt(512), positional encoding or input dropout.
A separate shared learned CLS vector is prepended using nn.Embedding(1,512). All three embedding tables use PyTorch default N(0,1) initialization.
Six independently initialized encoder layers, eight heads, FFN 2048, ReLU, internal dropout 0.1, post-norm epsilon 1e-5.
CLS -> Linear(512,1024) -> ReLU -> Linear(1024,2048) -> ReLU -> Linear(2048,vocabulary size).
No output activation or clipping. FP32 model, automatic attention backend. IDs are int64.
Only bin rounding and diagnostic norm reductions temporarily use double precision; model forward/backward, optimizer moments and targets remain FP32.
Adam beta=(0.9,0.98), epsilon=1e-9, no weight decay; Transformer schedule d^-0.5 min(update^-0.5, update*4000^-1.5).
Accumulate sample gradients in groups of 16, dividing by actual group size including the last group.

80/20 random run split, seed 42, validation count rounded upward. No donor grouping and no independent test set.
Training masks change by epoch; validation masks stay fixed. Mask floor(10% * reported count), minimum one.
Remove entire masked pairs; targets retain original decimal values represented as FP32. Loss selects output[species_id-1], averages masked errors within each sample, then averages samples.
One-entry samples use CLS alone when their entry is masked. Empty samples after excluding Unknown cause an explicit error.
All reported values are validated in [0,100]; invalid data cause an error rather than silent normalization, imputation or clamping.
Whole-number bins merge all reported values below 0.5 into category 0; this is the chosen input information loss.

## Files: where to change things

- checks.py: shared shape, ID-range and duplicate-target checks.
- settings.json / config.py: experiment settings and validation.
- data.py: TSV/gzip loading, reported-only arrays and vocabulary.
- split.py: random run split.
- embeddings.py: half-up bins and both embedding tables.
- masking.py: entry selection and pair removal.
- model.py: CLS, encoder, prediction head; encode() returns CLS for later analysis.
- losses.py: single shared masked-species loss implementation.
- train.py / validate.py: training and validation, respectively.
- runtime.py: seeds, attention backend and learning-rate schedule.
- checkpoints.py: atomic writes and strict resume compatibility.
- diagnostics.py: scalar monitoring, finite checks and replay tracing hooks.
- replay.py: replay suspicious update from recovery checkpoint.
- outputs.py: compressed history, one loss image and offline dashboard with all three plot types.
- tests/test_core.py: scientific correctness checks.

## Installation and first checks

Use the existing GPU environment with PyTorch 2.8. The requirements file documents dependencies; installing it can change package versions, so first check your existing environment:

    python -c "import torch,numpy,matplotlib; print(torch.__version__,torch.cuda.is_available())"
    python -m unittest discover -s tests -v
    OMP_NUM_THREADS=1 python tests/integration_cpu.py

For a missing plotting dependency:

    python -m pip install matplotlib

Edit settings.json to set input and output paths. The supplied input defaults to:
/mnt/data/shyam/nakulan/22_10/Encoder_only/non_asd_qc_pass_reported_required.tsv.gz

Run from this project folder:

    nohup python -u main.py --config settings.json --device cuda > training.log 2>&1 &
    tail -f training.log

A new run requires an empty output folder. This is a fresh architecture and does not load the old abundance-only model.

## Outputs and resume

With the default 100 epochs, the results folder contains 26 files, plus training.log outside it = 27 normal output files. Input data and source code are not included in this count. Captured failure/replay files are additional.

- dashboard.html: one offline dashboard with complete loss curves, an epoch selector for prediction/residual panels, exact epoch metrics and an analysis guide. Updated every ten epochs and at completion.
- loss.png: one shareable training/validation loss image, overwritten at each plotting interval.
- history.json.zlib: one compressed JSON file containing every epoch's metrics and all retained plotting-epoch prediction pairs (up to 10,000 pairs per plotting epoch, preserving the previous plotting cap). Read in Python with json.loads(zlib.decompress(Path('history.json.zlib').read_bytes())).
- experiment.json: resolved settings, species vocabulary, software/code/input fingerprints, train/validation run IDs and fixed validation masks.
- diagnostic_events.jsonl: compact monitoring records.
- recovery.pt: one rolling recovery checkpoint, overwritten at epoch start and every 250 updates.
- checkpoints/: 20 model/optimizer checkpoints, every five epochs through epoch 100.

All epochs contribute to loss history. Earlier prediction and residual panels remain accessible in the dashboard. The dashboard embeds images, so no internet or dashboard server is required. Combining outputs reduces clutter, not the storage needed by twenty independent model checkpoints.

    python -u main.py --config settings.json --device cuda --resume results_modular/checkpoints/epoch_0005.pt

Resume requires identical source code, resolved settings, input content, vocabulary and recorded environment. A changed architecture should start a new output directory. Resuming from recovery.pt also works at accumulation-group boundaries. Starting from an older checkpoint truncates later history records as epochs are rerun. The dashboard and loss image refresh at the next plotting interval; existing later checkpoint files remain until regenerated and must not be mistaken for newly resumed results.

## What happens on a numerical problem

No gradient clipping is used. Loss/predictions, gradients, parameters and optimizer tensors are checked. Large finite gradients also cause a diagnostic stop when the configurable maximum absolute accumulated gradient exceeds 1,000,000. This threshold is an engineering investigation trigger, NOT a scientifically universal limit, norm clipping, or parameter change.
Gradient checks run after every sample, so they have runtime overhead. Routine scalar events are emitted about every 63 updates. No all-epoch tensor history is retained.
failure.pt contains the sample/mask records for the current accumulation group and a short recent scalar history. It refers to recovery.pt, which holds the preceding clean training state. Stop and preserve this results folder before another run overwrites recovery.pt.
An interruption such as power loss may leave no failure.pt; use recovery.pt to resume.

    python -u replay.py --results results_modular --device cuda > replay.log 2>&1

Replay restores weights, Adam state and Python/CPU/CUDA random states, reconstructs order and masks and replays updates up to the affected group. It records layer-boundary forward/backward summaries and autograd-node gradient input/output summaries with maximum coordinates in replay_trace.jsonl. It does not change the source checkpoint or switch the attention backend. It stops before the suspect optimizer update.

Limits: automatic CUDA algorithms may not reproduce bitwise-identical results even with saved states. Source/environment checks help, but are not a guarantee. Fused attention may expose a single autograd node rather than QK/softmax/AV internals. Exact GPU instruction-level attribution requires further kernel-specific instrumentation. This project does not pretend that the first observed large gradient proves the underlying root cause. Trace indices are tensor coordinates; the preceding sample record maps visible token positions to species IDs (remember CLS is first).
The bundle saves failure metadata, not all parameter gradients or full activation arrays. Therefore it cannot retrospectively recover every intermediate value if replay fails to reproduce the event. More retention can be added as an explicitly chosen storage trade-off.
Replay hooks can alter scheduling and add overhead. A runtime error such as CUDA OOM may prevent detailed replay; address memory separately rather than interpreting it as a gradient cause.

## Before a full GPU run

Run the correctness tests and a short GPU pilot with a separate output directory, checking finite optimizer state and validation output. CPU verification does not establish A100 kernel stability or full-data performance.
Useful references: Python modules https://docs.python.org/3/tutorial/modules.html ; PyTorch autograd https://docs.pytorch.org/docs/2.8/autograd.html ; reproducibility https://docs.pytorch.org/docs/2.8/notes/randomness.html ; numerical embeddings https://arxiv.org/abs/2203.05556 .
