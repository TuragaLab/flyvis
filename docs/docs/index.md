<h1>
<p style="text-align:left;">
    <img id="flyvis-logo-light-content" src="images/flyvis_logo_light@150ppi.webp" width="50%" class="center">
    <img id="flyvis-logo-dark-content" src="images/flyvis_logo_dark@150ppi.webp" width="50%" class="center">
</p>
</h1>

# Flyvis Documentation

A connectome-constrained deep mechanistic network (DMN) model of the fruit fly visual system in PyTorch.

- Explore connectome-constrained models of the fruit fly visual system.
- Generate and test hypotheses about neural computations.
- Try pretrained models on your data.
- Develop custom models using our framework.

Flyvis is our official implementation of
[Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly visual system." Nature (2024).](https://www.nature.com/articles/s41586-024-07939-3)


## Usage Guide

To get started we recommend going through our tutorials. These will guide you through the core concepts and provide practical examples:

### Tutorials
1. [Explore the Connectome](examples/01_flyvision_connectome.md): Learn about the structure of the fly visual system connectome.
2. [Train the Network](examples/02_flyvision_optic_flow_task.md): Understand how to train the network on an optic flow task.
3. [Flash Responses](examples/03_flyvision_flash_responses.md): Explore how the model responses to flash stimuli.
4. [Moving Edge Responses](examples/04_flyvision_moving_edge_responses.md): Analyze the model's responses to moving edge stimuli.
5. [Ensemble Clustering](examples/05_flyvision_umap_and_clustering_models.md): Learn about clustering ensembles of models.
6. [Maximally Excitatory Stimuli](examples/06_flyvision_maximally_excitatory_stimuli.md): Discover how to find stimuli that maximally excite neurons.
7. [Custom Stimuli](examples/07_flyvision_providing_custom_stimuli.md): Learn how to provide your own custom stimuli to the model.

### Main Results

These notebooks show the main results of the paper:

1. [Fig. 1: Connectome-constrained and task-optimized models of the fly visual system.](examples/figure_01_fly_visual_system.md)
2. [Fig. 2: Ensembles of DMNs predict tuning properties.](examples/figure_02_simple_stimuli_responses.md)
3. [Fig. 3: Cluster analysis of DMN ensembles enables hypothesis generation and suggests experimental tests.](examples/figure_03_naturalistic_stimuli_responses.md)
4. [Fig. 4: Task-optimal DMNs largely recapitulate known mechanisms of motion computation.](examples/figure_04_mechanisms.md)

### API Reference

For detailed information about flyvis' components and functions, please refer to our [API Reference](reference/connectome.md) section. This includes documentation for key modules such as Connectomes, Network, NetworkView, and more.

### Scripts

We also provide a set of scripts for various tasks, including data download, training, validation, and analysis. You can start with the [Scripts](reference/scripts/scripts/download_pretrained_models.md) section of our documentation.
A good starting point is also the [pipeline manager](reference/scripts/scripts/pipeline_manager.md) to run the scripts in sequence on either LSF or SLURM compute clouds.

## Installation

### Quickstart with Google Colab

Try the models and code inside our Google Colab notebooks for a quickstart.

- [Explore the connectome](https://colab.research.google.com/drive/16xi96XS3whNhwMNeFihBNNgADVh60XHH?usp=sharing)
- [Provide custom stimuli](https://colab.research.google.com/drive/1xBJ-xLgmLGhXgkf8XLw2PRRlDrYQ1Hhv?usp=sharing)
- [Optic flow task]()
- [Flash responses]()
- [Moving edge responses]()
- [Umap and clustering]()
- [Maximally excitatory stimuli]()

### Local Installation

See [install.md](install.md) for details on how to install the package and download the pretrained models.

## Citation

```
@article{lappalainen2024connectome,
	title = {Connectome-constrained networks predict neural activity across the fly visual system},
	issn = {1476-4687},
	url = {https://doi.org/10.1038/s41586-024-07939-3},
	doi = {10.1038/s41586-024-07939-3},
	journal = {Nature},
	author = {Lappalainen, Janne K. and Tschopp, Fabian D. and Prakhya, Sridhama and McGill, Mason and Nern, Aljoscha and Shinomiya, Kazunori and Takemura, Shin-ya and Gruntman, Eyal and Macke, Jakob H. and Turaga, Srinivas C.},
	month = sep,
	year = {2024},
}
```

If you have any questions or encounter any issues, please check our [FAQ](faq.md) or [Contributing](contribute.md) pages for more information.
