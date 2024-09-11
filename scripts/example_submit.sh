# train an ensemble of 50 models
python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_training \
        --launch_validation \
        description='test' \
        train=true \
        +delete_if_exists=true

python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9998 \
        --task_name flow \
        --launch_training \
        --launch_validation \
        --launch_synthetic_recordings \
        description='test' \
        train=true \
        delete_if_exists=true \
        +task.original_split=true

python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9998 \
        --task_name flow \
        --launch_validation \
        --launch_synthetic_recordings \
        description='test' \
        train=true \
        delete_if_exists=true \
        +task.original_split=true

python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9998 \
        --task_name flow \
        --delete_recordings \
        --launch_synthetic_recordings \
        --launch_ensemble_analysis


python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9998 \
        --task_name flow \
        --launch_validation \
        --launch_synthetic_recordings \
        description='test' \
        train=true \
        delete_if_exists=true \
        +task.original_split=true

# validate an ensemble of 50 models
 python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_validation

# compute and store synthetic data
python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 0000 \
        --task_name flow \
        --launch_synthetic_recordings


python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_synthetic_recordings


python train_ensemble.py \
         --start 0 \
        --end 50  \
        --nP 8   \
        --gpu num=1   \
        --q gpu_l4  \
        --ensemble_id 9998   \
        --task_name flow   \
        --delete_umap_and_clustering \
        --launch_ensemble_analysis


python train_ensemble.py \
         --start 0 \
        --end 50  \
        --nP 8   \
        --gpu num=1   \
        --q gpu_l4  \
        --ensemble_id 0000  \
        --task_name flow   \
        --delete_recordings \
        --launch_synthetic_recordings

python train_ensemble.py \
         --start 0 \
        --end 50  \
        --nP 8   \
        --gpu num=1   \
        --q gpu_l4  \
        --ensemble_id 0000  \
        --task_name flow   \
        --launch_notebook_single \
        notebook_path=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/scripts/analysis/movingbar_currents_recording.ipynb
