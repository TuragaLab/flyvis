# Install

## Local installation

### Create virtual environment
#### Option 1: Using conda (recommended)

1. Create a new conda environment: `conda create --name flyvision python=3.9 -y`
2. Activate the new conda environment: `conda activate flyvision`

#### Option 2: Using venv

1. Create a new virtual environment: `python -m venv flyvision_env`
2. Activate the virtual environment:
   - On Windows: `flyvision_env\Scripts\activate`
   - On macOS and Linux: `source flyvision_env/bin/activate`

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
python scripts/download_pretrained_models.py
```

Make sure to run this command from your active flyvision environment.
