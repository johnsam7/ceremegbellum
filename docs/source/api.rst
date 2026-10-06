API reference
=============

The API is unstable and may change without notice.

Pipeline
--------

The main workflow has three stages: segment the subject's cerebellum, fit the
atlas surface to it, and build a combined cerebral and cerebellar source space
for MNE-Python.

.. currentmodule:: cmb

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_cerebellum_data
   segment_cerebellum
   create_cerebellar_surface
   setup_full_source_space
   

Plotting
--------

``plot_normal`` and ``plot_inflated`` require the ``viz`` extra (PyVista).
``plot_flatmap`` and ``plot_sagittal`` only need matplotlib.

.. currentmodule:: cmb.visualization

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_plot_data_from_stc
   plot_normal
   plot_inflated
   plot_flatmap
   plot_sagittal


Lower-level plotting helpers
----------------------------

These are called by :func:`get_plot_data_from_stc`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   morph_cerebellum_data
   morph_cortex_data


Utilities
---------

.. currentmodule:: cmb

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_subsampled_cerebellum_labels


Configuration
-------------

.. currentmodule:: cmb

.. py:data:: CMB_DATA_DIR

   Default directory for the atlas data and nnU-Net models, which is the
   package installation directory. Entry points accept a ``cmb_dir`` parameter
   to override it.

.. py:data:: __version__

   The installed package version.

