# Release Process

Releases are published to PyPI by the
[`Release` workflow](https://github.com/TuragaLab/flyvis/blob/main/.github/workflows/release.yml).
The manual steps further down are the fallback for when the workflow cannot be used.

## Automated release

Every push to `main` builds the source distribution and the wheel and checks them,
so a packaging mistake --- a data file that stopped being included, for instance ---
surfaces at merge time. Nothing is uploaded on a merge: PyPI versions are immutable
and can never be reused, so publishing is tied to a version tag rather than to every
commit that lands on `main`.

To cut a release:

1. Merge to `main` and let the tests and the build check pass.
2. Update `CHANGELOG.md` with the new version.
3. Tag and push:
   ```bash
   git tag -a v1.1.4 -F CHANGELOG.md
   git push origin main v1.1.4
   ```
4. Publish a GitHub Release pointing at that tag. That triggers the upload.

The workflow refuses to publish if the version derived by `setuptools_scm` does not
match the release tag, if the distributions fail `twine check`, or if the
precomputed response norms are missing from them. It then installs the built wheel
in a clean virtualenv and verifies that the constants can be read from the installed
package before uploading.

### One-time setup

Uploads use [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/), so
no API token is stored in the repository. On PyPI, under the `flyvis` project's
*Publishing* settings, add a GitHub publisher with:

| Field | Value |
| --- | --- |
| Owner | `TuragaLab` |
| Repository | `flyvis` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Adding required reviewers to the `pypi` environment in the repository settings makes
every upload wait for a human approval.

## Prerequisites

1. Ensure all tests pass:
```bash
pytest
```

2. Update and verify the documentation:
Follow the instructions in the [readme](README.md) to update and verify the documentation.
The deployment to github can be done last or via workflow.

3. Install required tools:
```bash
python -m pip install build twine
```

## Manual release steps

These are the fallback for when the automated release cannot be used.

0. **Test PyPi before committing (optional)**

One can create the wheel and upload it to Test PyPi before committing to develop the package.
This can be useful to test if the installation will work before commiting to a version number and
push to the remote repository. However, the wheels that are uploaded to Test PyPi cannot
be deleted so one should probably do this sparingly and not use version numbers one wants to reserve for
the actual release.

### Upload to Test PyPi
```bash
# Clean previous builds
rm -rf dist/

# Set version temporarily for this session manually (change version number)
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev7

# Now build and test
python -m build

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*
```

### Test Installation

In a clean environment, run these sporadic tests to verify the installation:
```bash
# Install in clean environment to test (change version number)
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ flyvis==0.0.0.dev7

# Test installation
flyvis download-pretrained
flyvis train --help
flyvis train-single --help
python -c "from flyvis import Network; n = Network()"
python -c "from flyvis import EnsembleView; e = EnsembleView('flow/0000')"


# Test custom configuration
flyvis init-config
# Add custom_syn_strength.yaml to flyvis_config/network/edge_config/syn_strength/
cat > flyvis_config/network/edge_config/syn_strength/custom_syn_strength.yaml << 'EOL'
defaults:
  - /network/edge_config/syn_strength/syn_strength@_here_
  - _self_

scale: 1.0
EOL
# Run training and check if custom config is loaded correctly
flyvis train-single --config-dir $(pwd)/flyvis_config network/edge_config/syn_strength=custom_syn_strength ensemble_and_network_id=0 task_name=flow delete_if_exists=true 2>&1 | while read line; do
    echo "$line"
    if [[ "$line" == *"'syn_strength': {'type': 'SynapseCountScaling'"* && "$line" == *"'scale': 1.0"* ]]; then
        echo "Found custom scale parameter in config!"
        pkill -P $$
        break
    fi
done
# Delete custom config
rm -rf flyvis_config

# Delete installation and downloaded models
pip uninstall flyvis -y

# When done testing, unset it
unset SETUPTOOLS_SCM_PRETEND_VERSION
```

### Commit Changes

Commit all open changes to the repository.

### Update Changelog

- Append entry in `CHANGELOG.md` with new version number
- Include all notable changes under appropriate sections, e.g.,
   - Breaking
   - Features
   - Documentation
   - Infrastructure
   - Distribution
   - Bug Fixes

```bash
git add CHANGELOG.md
git commit -m "docs: add changelog for v1.1.2"
```

### Create and Push Tag

```bash
# Create annotated tag using changelog
git tag -a v1.1.2 -F CHANGELOG.md

# Push to both remotes
git push origin main
git push origin v1.1.2
git push public_repo main
git push public_repo v1.1.2
```

### Build and Upload to PyPI
```bash
# Clean previous builds
rm -rf dist/

# Set version temporarily for this session manually
export SETUPTOOLS_SCM_PRETEND_VERSION=1.1.2

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### Create GitHub Release
   - Go to GitHub releases page
   - Create new release using the tag
   - Copy changelog entry into release description

## Post-release

1. Verify package can be installed from PyPI:
```bash
python -m pip install flyvis
```

## Check documentation is updated on the documentation website

## Version Numbering

We follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Notes

- Always test on Test PyPI before releasing to PyPI
- Ideally CI checks pass before releasing
- Keep both origin and public_repo remotes in sync
