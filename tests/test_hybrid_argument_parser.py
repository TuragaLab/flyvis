import tempfile

import pytest
import yaml

from flyvis.utils.config_utils import HybridArgumentParser


def test_basic_hybrid_args():
    """Test basic hybrid argument parsing with required args from CLI."""
    parser = HybridArgumentParser(
        hybrid_args={
            'task_name': {'required': True},
            'ensemble_id': {'required': True},
        }
    )

    # Test successful parsing with key=value format
    args = parser.parse_with_hybrid_args(['task_name=flow', 'ensemble_id=0045'])
    assert args.task_name == 'flow'
    assert args.ensemble_id == '0045'

    # Test missing required argument
    with pytest.raises(SystemExit):
        parser.parse_with_hybrid_args(['task_name=flow'])


def test_hybrid_args_with_types():
    """Test hybrid arguments with type specifications."""
    parser = HybridArgumentParser(
        hybrid_args={
            'ensemble_id': {'type': int, 'required': True},
            'task_name': {'type': str, 'required': True},
        }
    )

    args = parser.parse_with_hybrid_args([
        'ensemble_id=45',
        'task_name=flow',
    ])
    assert isinstance(args.ensemble_id, int)
    assert args.ensemble_id == 45
    assert isinstance(args.task_name, str)
    assert args.task_name == 'flow'


def test_mixed_arguments():
    """Test mixing standard argparse arguments with hybrid arguments."""
    parser = HybridArgumentParser(
        hybrid_args={
            'task_name': {'required': True},
            'ensemble_id': {'required': True},
        }
    )
    parser.add_argument('--gpu', type=str, default='num=1')
    parser.add_argument('--q', type=str, default='gpu_l4')

    args = parser.parse_with_hybrid_args([
        'task_name=flow',
        'ensemble_id=45',
        '--gpu=num=2',
        '--q=gpu_l8',
    ])
    assert args.task_name == 'flow'
    assert args.ensemble_id == '45'
    assert args.gpu == 'num=2'
    assert args.q == 'gpu_l8'


def test_hydra_style_config():
    """Test handling of Hydra-style configuration arguments."""
    parser = HybridArgumentParser(
        hybrid_args={
            'task_name': {'required': True},
            'ensemble_id': {'required': True},
        }
    )

    args = parser.parse_with_hybrid_args([
        'task_name=flow',
        'ensemble_id=45',
        'network.edge_config.syn_strength=0.5',
        'task.n_iters=2000',
    ])

    assert args.task_name == 'flow'
    assert args.ensemble_id == '45'
    assert hasattr(args, 'network.edge_config.syn_strength')
    assert hasattr(args, 'task.n_iters')
    assert getattr(args, 'network.edge_config.syn_strength') == '0.5'
    assert getattr(args, 'task.n_iters') == '2000'


def test_config_file_validation():
    """Test parsing arguments with a reference config file."""
    config = {
        "task_name": "default",
        "ensemble_id": "0000",
        "network": {"edge_config": {"syn_strength": 1.0}},
        "task": {"n_iters": 1000},
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
        yaml.dump(config, config_file)
        config_file.flush()

        parser = HybridArgumentParser(
            hybrid_args={
                'task_name': {'required': True},
                'ensemble_id': {'required': True},
            },
            drop_disjoint_from=config_file.name,
        )

        # Test that a warning is raised for invalid params
        with pytest.warns(
            UserWarning,
            match=r".*Argument.*invalid\.param=value.*does not affect the hydra config.*",
        ):
            args = parser.parse_with_hybrid_args([
                'task_name=flow',
                'ensemble_id=45',
                'network.edge_config.syn_strength=0.5',
                'task.n_iters=2000',
                'invalid.param=value',
            ])

        assert hasattr(args, 'task_name')
        assert args.task_name == 'flow'
        assert args.ensemble_id == '45'

        assert hasattr(args, 'network.edge_config.syn_strength')
        assert hasattr(args, 'task.n_iters')
        assert getattr(args, 'network.edge_config.syn_strength') == '0.5'
        assert getattr(args, 'task.n_iters') == '2000'

        assert not hasattr(args, 'invalid.param')


def test_unrecognized_arguments():
    """Test handling of unrecognized arguments."""
    parser = HybridArgumentParser(
        hybrid_args={'task_name': {'required': True}},
        allow_unrecognized=False,
    )

    with pytest.raises(SystemExit):
        parser.parse_args(['task_name=flow', 'unknown=value'])

    parser = HybridArgumentParser(
        hybrid_args={'task_name': {'required': True}},
        allow_unrecognized=True,
    )
    args = parser.parse_with_hybrid_args([
        'task_name=flow',
        'unknown=value',
    ])
    assert args.task_name == 'flow'
    assert hasattr(args, 'unknown')
    assert args.unknown == 'value'


def test_hydra_argv():
    """Test generation of Hydra-style argument list."""
    parser = HybridArgumentParser(
        hybrid_args={
            'task_name': {'required': True},
            'ensemble_id': {'required': True},
        }
    )

    # First parse arguments
    args = parser.parse_with_hybrid_args([
        'task_name=flow',
        'ensemble_id=45',
        'network.edge_config.syn_strength=0.5',
    ])

    # Convert the parsed args to hydra format
    hydra_args = [f"{key}={value}" for key, value in vars(args).items()]

    assert 'task_name=flow' in hydra_args
    assert 'ensemble_id=45' in hydra_args
    assert 'network.edge_config.syn_strength=0.5' in hydra_args
