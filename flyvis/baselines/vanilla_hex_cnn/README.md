# VanillaHexCNNBaseline in flyvis

This baseline integrates the legacy `dvs-sim` VanillaHexCNN training path into
`flyvis` as a separate baseline subpackage, without modifying the core
`MultiTaskSolver` pipeline.

## Files

- Model: `flyvis/baselines/vanilla_hex_cnn/models.py`
- Trainer: `flyvis/baselines/vanilla_hex_cnn/trainer.py`
- Hydra config:
  `flyvis/baselines/vanilla_hex_cnn/config/config.yaml`
- CLI entrypoint:
  `flyvis_cli/training/train_vanilla_hex_cnn.py`

## Results schema (partially EnsembleView-compatible)

Runs are written under:

- `results/baselines/<task_name>/<ensemble_id>/<model_id>/`

Each run records:

- `loss` (training curve)
- `validation/epe` (validation metric curve)
- `validation/loss` (same values as fallback)
- `chkpts/chkpt_<index>` checkpoint files
- `chkpt_index`, `chkpt_iter`, `best_chkpt_index`

This is sufficient for loading with `flyvis.network.EnsembleView` and plotting:

- `training_loss()`
- `validation_loss()`

Network-specific analyses are not guaranteed for this baseline.

## Command mapping from legacy to refactored flow

Legacy pattern:

```bash
./train_ensemble.py -start 0 -end 5 -id 0012 -task flow \
  -script train_vanilla_hex_cnn.py --train --hydra \
  arch=VanillaHexCNNBaseline arch_kwargs=L_structured arch_kwargs.n_frames=2
```

Refactored pattern:

```bash
flyvis train \
  --start 0 --end 5 --ensemble_id 12 --task_name flow \
  --train_script /abs/path/to/flyvis_cli/training/train_vanilla_hex_cnn.py \
  arch=VanillaHexCNNBaseline \
  arch_kwargs=L_structured \
  arch_kwargs.n_frames=2 \
  lr=0.1
```

## Sanity checks

### Forward shape check

```python
import torch
from flyvis.baselines.vanilla_hex_cnn.models import VanillaHexCNNBaseline

model = VanillaHexCNNBaseline(n_frames=2)
x = torch.randn(3, 2, 1, 721)
y = model(x)
assert y.shape == (3, 1, 2, 721)
```

### Short training smoke test

Use a tiny run (single process, few steps):

```bash
python flyvis_cli/training/train_vanilla_hex_cnn.py \
  ensemble_and_network_id=9999/000 \
  task_name=flow \
  arch=VanillaHexCNNBaseline \
  arch_kwargs=L_structured \
  arch_kwargs.n_frames=2 \
  epochs=1 \
  chkpt_every=1 \
  max_steps=2 \
  dataset.unittest=true \
  workers=0 \
  delete_if_exists=true
```

