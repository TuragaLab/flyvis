# Documentation

The documentation is available at: <https://turagalab.github.io/flyvis/>

## Building the Documentation

The documentation is built with [mkdocs](https://www.mkdocs.org/).

### Convert examples to markdown:

Only applies to making changes to the examples.

1. Run all notebooks inplace:

This takes a while. Make sure that recordings are precomputed and cached, i.e., run
`scripts/record.py` beforehand for the required ensemble.

```bash
export TESTING=true  # optional, will not run animations to speed things up
export JUPYTER_CONFIG_DIR=$(mktemp -d) # to avoid conflicts with notebook version and extensions

for notebook in ../examples/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$notebook" --inplace
done
```

2. Convert notebooks to markdown:

```bash
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir docs/examples/ --TagRemovePreprocessor.remove_cell_tags hide
```

3. Clear all notebook outputs (optional):
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ../examples/*.ipynb
```

### Build the scripts docs

Only applies to making changes to the scripts.

```bash
python script_to_md.py
```

## Serve the docs locally

```bash
mkdocs serve
```

## Deploy the docs to GitHub

See [mkdocs user guide](https://www.mkdocs.org/user-guide/deploying-your-docs/) for more details.

```bash
mkdocs gh-deploy
```
