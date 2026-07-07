# Cere-MEG-Bellum (CMB)

CMB is a Python package for **fitting a high-resolution cerebellar atlas to standard MRI (ARCUS)** and **MEG/EEG source space computation including the cerebellum**.

For more information about the method, please see:

> Samuelsson, J. G., J. D.Schmahmann, M. I.Sereno, B.Rosen, and M. S.Hämäläinen. 2026. “A Digital Anatomical Atlas of the Human Cerebellum at Subfolial Resolution.” Human Brain Mapping47, no. 4: e70497. https://doi.org/10.1002/hbm.70497.

## Features

- Automatic segmentation of cerebellar lobules using a trained nnU-Net model
- Diffeomorphic registration (ANTs SyNCC) of a high-resolution cerebellar template to subject anatomy
- Construction of cerebellar cortical source spaces compatible with MNE-Python
- Visualization of cerebellar data in normal, inflated, and flatmap views

## Requirements

- Python >= 3.10
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (for MRI conversion and recon-all)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (installed automatically as a dependency)

## Installation

1. Install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) for your system.

2. Clone this repository and navigate to project root.

   ```bash
   git clone https://github.com/johnsam7/ceremegbellum.git
   cd ceremegbellum
   ```

3. Create a new virtual environment for Cere-MEG-Bellum and activate it.
   You can do this, for example, with [uv](https://docs.astral.sh/uv/) (recommended!),
   [venv](https://docs.python.org/3/library/venv.html) or [conda](https://github.com/conda/conda).
   `uv` and `conda` are good options because they can also manage the Python version without
   depending on system Python.

   ```bash
   # uv
   uv venv cmb-env --python 3.12 --seed
   source cmb-env/bin/activate
   # conda
   conda create -n cmb-env python=3.12 pip
   conda activate cmb-env
   # venv
   python -m venv cmb-env
   source cmb-env/bin/activate
   ```

4. Install correct PyTorch version for your hardware. Look up the exact installation on [PyTorch webpage](https://pytorch.org/get-started/locally/).
   Here is **example** command for hardware with NVIDIA GPU with CUDA 12.6 support.

   ```bash
   # NOTE: PyTorch website uses pip3 in the command.
   # Inside virtual environment pip and pip3 are the same.
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

   # Or use uv for faster installation!
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

5. Install Cere-MEG-Bellum with rest of the dependencies.

   ```bash
   pip install ".[viz]"

   # Or use uv for faster installation!
   uv pip install ".[viz]"
   ```

   This installs the core package plus [PyVista](https://docs.pyvista.org/) for 3D visualization. If you don't need 3D views (normal/inflated) and only want flatmaps, you can use `pip install .` instead.

   If you are installing the package for development, add `-e` flag to the installation command to make
   the changes to the source code immediately reflect to the installed package. For example: `uv pip install -e ".[viz]"`.

**Note:** On headless systems (no display), plots are automatically saved as PNG files. On **remote desktops** (e.g., NoMachine, VNC),
if 3D views segfault, unset DISPLAY before importing CMB to force offscreen rendering:

```python
import os
os.environ.pop('DISPLAY', None)
```

## Quick Start

See [`examples/example_script.py`](examples/example_script.py) for a complete end-to-end example using the MNE sample dataset.

## License

CMB is licensed under the [MIT License](LICENSE).

Copyright (c) 2021-2026, authors of CMB. All rights reserved.
