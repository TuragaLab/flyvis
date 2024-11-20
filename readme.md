<h1>
<p style="text-align:left;">
    <img src="docs/docs/images/flyvis_logo_light@150ppi.webp" width="50%" alt="Flyvis Logo">
</p>
</h1>

A connectome-constrained deep mechanistic network (DMN) model of the fruit fly visual system in PyTorch.

- Explore connectome-constrained models of the fruit fly visual system.
- Generate and test hypotheses about neural computations.
- Try pretrained models on your data.
- Develop custom models using our framework.

Flyvis is our official implementation of
[Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly visual system." Nature (2024).](https://www.nature.com/articles/s41586-024-07939-3)

## Documentation

For detailed documentation, installation instructions, tutorials, and API reference, visit our [documentation website](https://turagalab.github.io/flyvis/).

## Tutorials

Explore our tutorials to get started with flyvis. You can read the prerun tutorials in the docs or try them yourself for a quick start in Google Colab:

1. Explore the Connectome
   - [Tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/01_flyvision_connectome.ipynb)

2. Train the Network on the Optic Flow Task
   - [Tutorial](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/02_flyvision_optic_flow_task.ipynb)

3. Flash Responses
   - [Tutorial](https://turagalab.github.io/flyvis/examples/03_flyvision_flash_responses/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/03_flyvision_flash_responses.ipynb)

4. Moving Edge Responses
   - [Tutorial](https://turagalab.github.io/flyvis/examples/04_flyvision_moving_edge_responses/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/04_flyvision_moving_edge_responses.ipynb)

5. Ensemble Clustering
   - [Tutorial](https://turagalab.github.io/flyvis/examples/05_flyvision_umap_and_clustering_models/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/05_flyvision_umap_and_clustering_models.ipynb)

6. Maximally Excitatory Stimuli
   - [Tutorial](https://turagalab.github.io/flyvis/examples/06_flyvision_maximally_excitatory_stimuli/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/06_flyvision_maximally_excitatory_stimuli.ipynb)

7. Custom Stimuli
   - [Tutorial](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/)
   - [Google Colab](https://colab.research.google.com/github/TuragaLab/flyvis/blob/main/examples/07_flyvision_providing_custom_stimuli.ipynb)


## Main Results

Find the notebooks for the main results in the documentation.

- [Fig. 1: Connectome-constrained and task-optimized models of the fly visual system](https://turagalab.github.io/flyvis/examples/figure_01_fly_visual_system/)
- [Fig. 2: Ensembles of DMNs predict tuning properties](https://turagalab.github.io/flyvis/examples/figure_02_simple_stimuli_responses/)
- [Fig. 3: Cluster analysis of DMN ensembles enables hypothesis generation and suggests experimental tests](https://turagalab.github.io/flyvis/examples/figure_03_naturalistic_stimuli_responses/)
- [Fig. 4: Task-optimal DMNs largely recapitulate known mechanisms of motion computation](https://turagalab.github.io/flyvis/examples/figure_04_mechanisms/)

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

## Links

- [Nature Article](https://www.nature.com/articles/s41586-024-07939-3)
- [Documentation](https://turagalab.github.io/flyvis/)

## Correspondence

For questions or inquiries, [please contact us.](mailto:janne.lappalainen@uni-tuebingen.de?cc=jakob.macke@uni-tuebingen.de,turagas@janelia.hhmi.org)
