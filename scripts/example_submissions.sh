#!/bin/bash

# See the documentation under https://www.turagalab.github.io/flyvis for details of script usage
# and command line arguments.

# Example 1: Full pipeline for an ensemble 0001 on the flow task with defaults
python pipeline_manager.py \
    --command train validate record analysis notebook_per_model notebook_per_ensemble \
    --ensemble_id 0001 \
    --task_name flow


# Example 2: Run analysis and generate notebooks for an existing ensemble 0003 with defaults
python pipeline_manager.py \
    --command analysis notebook_per_model notebook_per_ensemble \
    --ensemble_id 0003 \
    --task_name flow

# Example 3: Full pipeline for ensemble 0004 on the flow task, with custom GPU and queue settings
python pipeline_manager.py \
    --command train validate record analysis notebook_per_model notebook_per_ensemble \
    --ensemble_id 0004 \
    --task_name flow \
    --gpu "num=2" \
    --q gpu_l8

# Example 4: Run only the record and analysis steps for existing ensemble 0005 on the depth task, with a dry run
python pipeline_manager.py \
    --command record analysis \
    --ensemble_id 0005 \
    --task_name depth \
    --dry

# Example 5: Full pipeline for ensemble 0006 on the flow task, with custom training settings
python pipeline_manager.py \
    --command train validate record analysis notebook_per_model notebook_per_ensemble \
    --ensemble_id 0006 \
    --task_name flow \
    --start 0 \
    --end 100 \
    --nP 8
