# Install

## Local installation

1. create a new conda environment `conda create --name flyvision -y`
2. activate the new conda environment `conda activate flyvision`
3. install python `conda install "python>=3.7.11,<3.10.0"`
4. clone the repository `git clone https://github.com/TuragaLab/flyvis.git`
5. navigate to the repo `cd flyvis` and install in developer mode `pip install -e .`

## Download pretrained models

1. run `python scripts/download_pretrained_models.py` from active conda environment