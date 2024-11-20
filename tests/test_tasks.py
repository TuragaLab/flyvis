from datamate import Namespace, set_root_context

from flyvis.task.tasks import Task


def test_task(mock_sintel_data, connectome, tmp_path_factory):
    task_config = Namespace(
        type="Task",
        dataset=Namespace(
            type="MultiTaskSintel",
            tasks=["flow"],
            boxfilter=dict(extent=15, kernel_size=13),
            vertical_splits=3,
            n_frames=19,
            center_crop_fraction=0.7,
            dt=1 / 50,
            augment=True,
            random_temporal_crop=True,
            all_frames=False,
            resampling=True,
            interpolate=True,
            p_flip=0.5,
            p_rot=5 / 6,
            contrast_std=0.2,
            brightness_std=0.1,
            gaussian_white_noise=0.08,
            gamma_std=None,
            _init_cache=True,
            unittest=True,
            flip_axes=[0, 1, 2, 3],
            sintel_path=str(mock_sintel_data),
        ),
        decoder=Namespace(
            flow=Namespace(
                type="DecoderGAVP",
                shape=[8, 2],
                kernel_size=5,
                const_weight=0.001,
                n_out_features=None,
                p_dropout=0.5,
            )
        ),
        loss=Namespace(flow="l2norm"),
        batch_size=4,
        num_workers=0,
        n_iters=250000,
        train_ratio=None,
        seed=1,
        transforms=None,
        train_seq_ind=None,
        val_seq_ind=None,
    )
    with set_root_context(tmp_path_factory.mktemp("tmp")):
        task = Task(
            task_config.dataset,
            task_config.decoder,
            task_config.loss,
            batch_size=4,
            n_iters=250_000,
            n_folds=4,
            fold=1,
            seed=0,
        )
        assert task is not None
        assert task.dataset is not None

        decoder = task.init_decoder(connectome)
        assert decoder is not None
