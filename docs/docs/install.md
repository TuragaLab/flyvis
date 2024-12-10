# Install

## Local installation

Make sure to run `flyvis` in your active python environment.

### Option 1: Install from PyPI

```bash
pip install flyvis
```

> `flyvis` is tested on Linux for python 3.9, 3.10, 3.11, 3.12.

### Option 2: Install from source

1. Clone the repository: `git clone https://github.com/TuragaLab/flyvis.git`
2. Navigate to the repo: `cd flyvis`
3. Install in developer mode: `pip install -e .`

For development, documentation, or to run examples, you can install additional dependencies:
- `pip install -e ".[dev,examples,docs]"`


## Download pretrained models

After installation, download the pretrained models by running:

```
flyvis download-pretrained
```

## Set environment variables

`flyvis` uses environment variables for configuration. These can be set using a `.env` file or manually.

### Option 1: Using .env file

Create a `.env` file in one of these locations (in order of precedence):

1. Your current working directory (where you run your scripts)
2. The root directory of your project
3. Any parent directory of your working directory

Example `.env` file contents:
```bash
# Set your preferred data directory (default: './data' relative to package directory)
FLYVIS_ROOT_DIR='path/to/your/data'
```

### Option 2: Setting environment variables manually

You can also set the environment variables directly in your shell:

```bash
export FLYVIS_ROOT_DIR='path/to/your/data'
```

> Note: The data directory is where `flyvis` will store downloaded models, cached data, and other files. If `FLYVIS_ROOT_DIR` is not set, it defaults to a `data` folder in the package directory.

## Training new models

For training new models with custom settings, the following command will create a default config in your current working directory to be able to create overrides.

```bash
flyvis init-config
```

See the config [hydra config docs](reference/hydra_default_config.md) for more details.
