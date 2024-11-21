import argparse
import shutil
from importlib.resources import files
from pathlib import Path


def get_valid_config_groups(pkg_config_dir):
    """Get list of valid config groups from package config directory."""
    config_path = Path(pkg_config_dir)
    valid_groups = []

    for path in config_path.rglob('*.yaml'):
        relative_path = path.relative_to(config_path)
        group = str(relative_path.parent / relative_path.stem)
        valid_groups.append(group)

    return sorted(valid_groups)


def init_config(args):
    """Initialize a local config directory with defaults from the package."""
    pkg_config_dir = str(files('flyvis') / 'config')
    output_path = Path(args.output_dir).resolve()
    success = False

    valid_groups = get_valid_config_groups(pkg_config_dir)

    if not args.config_groups:
        # Copy entire config structure contents to flyvis_config
        config_path = output_path / 'flyvis_config'
        config_path.mkdir(parents=True, exist_ok=True)

        # Copy contents of pkg_config_dir to flyvis_config
        for item in Path(pkg_config_dir).iterdir():
            dst = config_path / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            else:
                shutil.copytree(item, dst, dirs_exist_ok=True)

        print(f"Copied full config structure to {config_path.absolute()}")
        print(
            "Remember: When creating custom configs, use different names than the "
            "defaults."
        )
        success = True
    else:
        # Copy only specified config groups
        for group in args.config_groups:
            # Strip leading 'config/' if present
            group = group[7:] if group.startswith('config/') else group
            src_path = Path(pkg_config_dir) / group
            if not src_path.exists():
                print(f"Warning: Config group '{group}' not found in package")
                print("\nValid config groups:")
                for valid_group in valid_groups:
                    print(f"  - {valid_group}")
                continue

            dst_path = output_path / group
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

            print(f"Copied {group} to {dst_path.absolute()}.")
            print(
                "Remember: When creating custom configs, use different names than the "
                "defaults"
            )
            success = True

    if not success:
        print("\nNo valid config groups were copied. Available config groups:")
        for group in valid_groups:
            print(f"  - {group}")
        raise ValueError("No valid config groups were found to copy")

    # Update README content with correct path and location
    config_dir = '$(pwd)/flyvis_config' if not args.config_groups else '$(pwd)'
    readme_path = (
        output_path / ('flyvis_config' if not args.config_groups else '') / 'README.md'
    )
    readme_content = f"""# Custom FlyVis Configuration

This directory contains custom configurations for FlyVis. To use these configs:

Important: When creating custom configs, use different names than the defaults to
ensure proper override behavior.
For example:
- Default: syn_strength.yaml
- Custom: custom_syn_strength.yaml

Override specific configs:
```bash
flyvis train-single network/edge_config/syn_strength=custom_syn_strength --config-dir \
{config_dir}
```

The directory structure mirrors the package config structure, allowing you to override
specific parts while keeping others at their defaults.
Note that Hydra will look in the package's default config directory first, so your
custom configs must have different names.
"""

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(
        f"\nConfig initialization complete! See {readme_path.absolute()} for usage "
        "instructions."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Initialize a local config directory with defaults from the package'
    )
    parser.add_argument(
        '--output-dir', default='.', help='Directory where config should be initialized'
    )
    parser.add_argument(
        '--config-groups',
        nargs='*',
        help=(
            'Specific config groups to initialize '
            '(e.g., network/edge_config/syn_strength). If none specified, copies all.'
        ),
    )

    args = parser.parse_args()
    init_config(args)
