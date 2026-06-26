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

### Standard Installation

On most systems with a modern toolchain (GCC >= 9.3):

```bash
git clone https://github.com/johnsam7/ceremegbellum.git
cd ceremegbellum
pip install -e ".[viz]"
```

This installs the core package plus [PyVista](https://docs.pyvista.org/) for 3D visualization. If you don't need 3D views (normal/inflated) and only want flatmaps, you can use `pip install -e .` instead.

**Note:** On headless systems (no display), plots are automatically saved as PNG files. On **remote desktops** (e.g., NoMachine, VNC), if 3D views segfault, unset DISPLAY before importing CMB to force offscreen rendering:

```python
import os
os.environ.pop('DISPLAY', None)
```

### Installation on Systems with Older Toolchains

Some institutional clusters (e.g., CentOS 7 / RHEL 7) ship with GCC < 9.3 and cannot compile packages like NumPy or SciPy from source. If `pip install` fails with **"NumPy requires GCC >= 9.3"**, use the following approach:

1. **Get a supported Python version (3.12 recommended).** If conda is available, create a new environment. Otherwise, download a standalone build from [python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases) (look for `cpython-3.12.*-x86_64-unknown-linux-gnu-install_only.tar.gz`) and create a venv:

   ```bash
   # Extract standalone Python 3.12
   tar xzf cpython-3.12*-x86_64-unknown-linux-gnu-install_only.tar.gz
   /path/to/python/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. **Install compiled dependencies as pre-built wheels:**

   ```bash
   pip install --only-binary :all: \
       numpy scipy pandas matplotlib torch antspyx \
       scikit-image scikit-learn imagecodecs numexpr blosc2 \
       Pillow connected-components-3d SimpleITK \
       "timm<1.0.23" torchvision einops seaborn dicom2nifti
   ```

3. **Install source-only packages (pure Python, no compiler needed):**

   ```bash
   pip install --no-deps nnunet acvl-utils dynamic-network-architectures \
       batchgenerators batchgeneratorsv2 ndindex graphviz yacs \
       fft-conv-pytorch future medpy
   ```

4. **Install CMB and remaining dependencies:**

   ```bash
   pip install --only-binary :all: mne nibabel pooch
   pip install --no-deps -e .
   ```

5. **Verify the installation:**

   ```bash
   python -c "import cmb; print(cmb.__version__)"
   ```

## Quick Start

See [`examples/example_script.py`](examples/example_script.py) for a complete end-to-end example using the MNE sample dataset.

## License

CMB is licensed under the [MIT License](LICENSE).

Copyright (c) 2021-2026, authors of CMB. All rights reserved.
