python launch_ensemble.py \
        --start 0 \
        --end 5 \
        --launch_training \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --script train.py \
        description='test' \
        train=true \
        +delete_if_exists=true 