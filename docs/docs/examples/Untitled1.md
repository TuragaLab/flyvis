```python
from typing import List
from tqdm import tqdm
import torch
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

from pathlib import Path
from datamate import root, Directory

import flyvision
from flyvision import NetworkView, results_dir
from flyvision.datasets.sintel import MultiTaskSintel
from flyvision.utils.activity_utils import LayerActivity

%load_ext autoreload
%autoreload 2
```


```python
nnview = NetworkView(flyvision.results_dir / "opticflow/000/0000")
```


```python
flynet = nnview.init_network()
```


```python
decoder = nnview.init_decoder()
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    Input In [4], in <cell line: 1>()
    ----> 1 decoder = nnview.init_decoder()


    File ~/projects/flyvision/flyvision/network.py:898, in NetworkView.init_decoder(self, chkpt, decoder)
        888 def init_decoder(self, chkpt="best_chkpt", decoder=None):
        889     """Initialize the decoder.
        890 
        891     Args:
       (...)
        896         decoder instance.
        897     """
    --> 898     raise NotImplementedError("Decoder initialization not implemented yet.")
        899     if self._initialized["decoder"] and decoder is None:
        900         return self.decoder


    NotImplementedError: Decoder initialization not implemented yet.



```python
dataset = MultiTaskSintel()
```


```python
dataset.augment = False
```


```python
dataset.dt = 1/50
```


```python
state = flynet.fade_in_state(1.0, 1/50, dataset[0]["lum"][0])
```


```python
flynet.stimulus.add_input(dataset[0]["lum"][None])
```


```python
activity = flynet(flynet.stimulus(), dataset.dt, state=state)
```


```python
flow = decoder(activity)
```


```python
flow.shape
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [10], in <cell line: 1>()
    ----> 1 flow.shape


    NameError: name 'flow' is not defined



```python
anim = flyvision.animations.sintel.SintelSample(dataset[0]["lum"][None], dataset[0]["flow"][None], flow)
anim.animate_in_notebook()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [13], in <cell line: 1>()
    ----> 1 anim = flyvision.animations.sintel.SintelSample(dataset[0]["lum"][None], dataset[0]["flow"][None], flow)
          2 anim.animate_in_notebook()


    AttributeError: module 'flyvision.animations' has no attribute 'sintel'



```python

```
