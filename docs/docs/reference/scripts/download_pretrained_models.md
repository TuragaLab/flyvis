# Download Pretrained Models

`download_pretrained_models.py`

```
usage: download_pretrained_models.py [-h] [--skip_large_files]

Download pretrained models and UMAP clustering results. This script downloads two ZIP files from Google Drive:
1. results_pretrained_models.zip: Contains pretrained neural networks.
2. results_umap_and_clustering.zip: Contains UMAP embeddings and clustering.
The files are downloaded and unpacked to the 'data' directory in the project root.

optional arguments:
  -h, --help          show this help message and exit
  --skip_large_files  Skip downloading large files. If set, only 'results_pretrained_models.zip' will be downloaded, and 'results_umap_and_clustering.zip' will be skipped.

```
