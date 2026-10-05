API reference
=============

The API is unstable and may change without notice.

Pipeline
--------

The main workflow has three stages: segment the subject's cerebellum, fit the
atlas surface to it, and build a combined cerebral and cerebellar source space
for MNE-Python. :func:`~cmb.get_cerebellum_data` downloads the atlas and the
nnU-Net models needed by the first two stages.

.. currentmodule:: cmb

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_cerebellum_data
   segment_cerebellum
   create_cerebellar_surface
   setup_full_source_space

Preparing data for plotting
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_plot_data_from_stc

.. currentmodule:: cmb.functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_subsampled_cerebellum_labels

.. currentmodule:: cmb.visualization

.. autosummary::
   :toctree: generated/
   :nosignatures:

   morph_cerebellum_data
   morph_cortex_data
   interpolate_cerebellum_data

Plotting
--------

``plot_normal`` and ``plot_inflated`` require the ``viz`` extra (PyVista).
``plot_flatmap`` and ``plot_sagittal`` only need matplotlib.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_normal
   plot_inflated
   plot_flatmap
   plot_sagittal

Configuration
-------------

.. currentmodule:: cmb

.. py:data:: CMB_DATA_DIR

   Default directory for the atlas data and nnU-Net models, which is the
   package installation directory. Every entry point accepts a ``cmb_dir`` or
   ``cmb_path`` argument to override it, for example when the package directory
   is not writable.

.. py:data:: __version__

   The installed package version.

Advanced
--------

The following is only needed when working with custom meshes directly.

.. currentmodule:: cmb.source_space

.. autosummary::
   :toctree: generated/
   :nosignatures:

   calculate_normals
