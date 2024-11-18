# Install

## Local installation

> Note: `flyvis` is only tested on Linux.

### Create virtual environment
#### Option 1: Using conda (recommended)

1. Create a new conda environment: `conda create --name flyvision python=3.9 -y`
2. Activate the new conda environment: `conda activate flyvision`

#### Option 2: Using venv

1. Create a new virtual environment: `python -m venv flyvision_env`
2. Activate the virtual environment: `source flyvision_env/bin/activate`

### Clone repo and install

3. Clone the repository: `git clone https://github.com/TuragaLab/flyvis.git`
4. Navigate to the repo: `cd flyvis`
5. Install in developer mode: `pip install -e .`

For development, documentation, or to run examples, you can install additional dependencies:
- For development: `pip install -e ".[dev]"`
- For documentation: `pip install -e ".[docs]"`
- For examples: `pip install -e ".[example]"`

> Note: We make flyvision available on pypi soon to simplify the installation process.

## Download pretrained models

After installation, download the pretrained models by running:

```
flyvis download-pretrained
```

Make sure to run this command from your active flyvision environment.


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
