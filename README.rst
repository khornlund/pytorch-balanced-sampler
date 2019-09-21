========================
pytorch_balanced_sampler
========================

PyTorch implementations of `BatchSampler` that under/over sample according to a chosen parameter
alpha, in order to create a balanced training distribution.

Usage
=====

Installation
------------
.. code-block:: bash

  $ conda env create --file environment.yaml
  $ conda activate pytorch_balanced_sampler

Tests
-----
.. code-block:: bash

  $ pytest tests

Authors
=======
`pytorch_balanced_sampler` was written by `Karl Hornlund <karlhornlund@gmail.com>`_.
