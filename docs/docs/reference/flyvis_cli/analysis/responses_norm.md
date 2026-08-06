# Store Response Normalization Constants


::: flyvis_cli.analysis.responses_norm
    options:
      heading_level: 4


```
usage:
flyvis responses-norm [-h] task_name=TASK ensemble_id=XXXX [options]
       or
responses_norm.py [-h] task_name=TASK ensemble_id=XXXX [options]

Examples:
    Store the constants of ensemble flow/0000 next to the ensemble:
    flyvis responses-norm task_name=flow ensemble_id=0000

    Additionally write them into the constants shipped with the package:
    flyvis responses-norm task_name=flow ensemble_id=0000 --export

Compute the normalization constants of an ensemble from its responses to
naturalistic stimuli and store them in the ensemble directory, so that they never
have to be recomputed. Responses that are not cached yet are simulated, which is
the expensive part of this script.

options:
  -h, --help            show this help message and exit
  --validation_subdir VALIDATION_SUBDIR
  --loss_file_name LOSS_FILE_NAME
  --force               Recompute even if constants are already stored.
  --export              Also write the constants into the file shipped with the
                        package. For maintainers preparing a release of an ensemble.

Hybrid Arguments:
  --task_name TASK_NAME
                        task_name=value: Name of the task, e.g. 'flow'. (Required)
  --ensemble_id ENSEMBLE_ID
                        ensemble_id=value: Id of the ensemble, e.g. '0000'. (Required)

```

## Where the constants are looked up

`Ensemble.responses_norm` resolves the constants in this order and only simulates
naturalistic stimuli responses if none of them apply:

1. `<ensemble_dir>/responses_norm.h5` --- written whenever constants are computed
   for an ensemble, so a custom ensemble pays the cost only once.
2. `flyvis/data/responses_norm.h5` --- constants for the released ensembles, shipped
   with the package and loaded silently, so nothing has to be downloaded or
   simulated to reproduce the paper figures.

## How the right constant is matched to the right model

Constants are stored per model name and per checkpoint, never per position:

- **Order does not matter.** Stored constants are looked up by model name and
  returned in the order of `ensemble.names`, which is the order in which the
  ensemble's responses are concatenated along `network_id`. An ensemble that was
  never sorted, was sorted by validation error, or was built from an arbitrary list
  of model paths all get their constants in their own order.
- **The checkpoint has to be the same one.** Each constant records the SHA256 of
  the checkpoint its responses were computed from, and stored constants are only
  used if that hash matches the checkpoint the ensemble currently resolves to.
  Comparing file names alone would not be enough: within one ensemble every model's
  best checkpoint is typically called `chkpt_00000`, so a name carries almost no
  information, and a checkpoint retrained in place --- or a different ensemble
  trained into a directory of the same name --- would silently pick up foreign
  constants. If the hashes do not match, the constants are recomputed instead.
  Hashing a 50-model ensemble takes about 60 ms.
- **Constants written before hashes were recorded** are still read, and fall back to
  comparing checkpoint file names.
