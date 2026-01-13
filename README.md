# SPECULA
Python AO end-to-end simulator

SPECULA is a Python-based, object-oriented software derived from [PASSATA](https://arxiv.org/abs/1607.07624) and developed
by the Adaptive Optics group at the Arcetri Observatory for end-to-end Monte-Carlo simulations of adaptive optics systems.
It can be accelerated using GPU-CUDA via CuPy.

See the documentation here: [specula.readthedocs.io](https://specula.readthedocs.io/en/latest/)

## Directories

- **docs**: contains the documentation.
- **config**: contains functions and parameter files to calibrate and run a closed loop of an adaptive optics system (single-conjugated, multi-conjugated, natural, laser, ...).
- **specula**: the main library, structured as follows:
  - **data**: data required by utility functions.
  - **data_objects**: classes that wrap the data and provide methods to access them.
  - **display**: classes for data visualization.
  - **lib**: utility functions used by multiple objects.
  - **processing_objects**: classes that model the simulation elements as a function of inputs and time.
  - **scripts**: various scripts.
- **test**: contains functions to test SPECULA using the `unittest` framework.

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- flask
- flask-socketio
- socketio
- scikit-image (for physical propagation)
- cupy (for GPU acceleration, optional)

### Optional libraries

Some features require additional libraries:
- **orthogram**: for automatic block diagram creation (see [orthogram](https://pypi.org/project/orthogram/), requires **pycairo**)
- **control**: for conversion of transfer function system in SPECULA format and vice-versa and analysis of transfer function

## Contributing to SPECULA
To contribute to SPECULA, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`
3. Make your changes and **add tests for the new functionality.**
4. Commit your changes: `git commit -m '<commit_message>'`
5. Push to the branch: `git push`
6. Create the pull request.

We require tests for all new features to ensure the stability of the project.

## Citation

If you use SPECULA in your research, please cite:

```bibtex
@article{specula2026,
  author = {Fabio Rossi and Alfio Puglisi and Guido Agapito},
  title = {{Introducing a new generation adaptive optics simulation framework: from PASSATA to SPECULA}},
  volume = {12},
  journal = {Journal of Astronomical Telescopes, Instruments, and Systems},
  number = {1},
  publisher = {SPIE},
  pages = {019001},
  year = {2026},
  doi = {10.1117/1.JATIS.12.1.019001},
  URL = {https://doi.org/10.1117/1.JATIS.12.1.019001}
}
```
