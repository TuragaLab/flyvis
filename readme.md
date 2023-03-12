# Connectome-constrained deep mechanistic networks predict neural responses across the fly visual system at single-neuron resolution

Janne K. Lappalainen,<sup>1,2</sup> Fabian D. Tschopp,<sup>2</sup> Sridhama Prakhya,<sup>2</sup> Mason McGill,<sup>2,3</sup>
Aljoscha Nern,<sup>2</sup> Kazunori Shinomiya,<sup>2</sup> Shin-ya Takemura,<sup>2</sup> Eyal Gruntman,<sup>2</sup>
Jakob H. Macke,<sup>1,4</sup> Srinivas C. Turaga<sup>2∗</sup>


<sup>1</sup>Machine Learning in Science, Tübingen University and Tübingen AI Center, Germany
<sup>2</sup>HHMI Janelia Research Campus, Ashburn, VA, USA
<sup>3</sup>Computation and Neural Systems, California Institute of Technology, Pasadena, CA, USA
<sup>4</sup>Max Planck Institute for Intelligent Systems, Tübingen, Germany
<sup>∗</sup>turagas@janelia.hhmi.org.


We release our connectome, task, and single-neuron dynamics constrained deep mechanistic network models of the Drosophila visual system.

# Examples

The following example notebooks show how to use our library.

- 01_flyvision_connectome.ipynb to visualize connectivity
- 02_flyvision_optic_flow_task.ipynb to compute and visualize optic flow (tbd)
- 03_flyvision_flash_responses.ipynb to compute and visualize flash responses (tbd)
- 04_flyvision_moving_edge_responses.ipynb to compute and visualize moving edge responses (tbd)
- 05_flyvision_naturalistic_stimuli_responses.ipynb to compute and visualize responses to naturalistic stimuli (tbd)
- 06_flyvision_tmy_predictions.ipynb to compute and visualize TmY3 responses (tbd)
- 07_flyvision_providing_custom_stimuli.ipynb to compute and visualize responses to your own stimuli

# Installation

1. clone the repository `git clone https://github.com/TuragaLab/flyvis.git`
2. make sure conda is installed
3. create a new conda environment `conda create --name flyvision -y`
4. activate the new conda environment `conda activate flyvision`
5. install python `conda install "python>=3.7.11,<3.10.0"`
6. navigate to the repository and install in developer mode `pip install -e .`
7. install pytorch, torchvision, cuda `conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3.1 -c pytorch`
8. run `pytest`


# Help and support
lappalainenjk@gmail.com

# Acknowledgements


# Citation
```
@article{lappalainen2023connectome,
  title={Connectome-constrained deep mechanistic networks predict neural
  responses across the fly visual system at single-neuron resolution},
  author={Lappalainen, Janne K and Tschopp, Fabian D and Prakhya, Sridhama and
  McGill, Mason and Nern, Aljoscha and Shinomiya, Kazunori and Takemura, Shin-ya
   and Gruntman, Eyal and Macke, Jakob H and Turaga, Srinivas C},
  journal={bioRxiv},
  year={2023}
}
```

# License

# Contributing

