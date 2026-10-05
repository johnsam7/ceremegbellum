API reference
=============

The API is unstable and may change without notice.

Pipeline
--------

.. currentmodule:: cmb

The main workflow has three stages: segment the subject's cerebellum
(:func:`segment_cerebellum`), fit the atlas surface to it
(:func:`create_cerebellar_surface`), and build a combined cerebral and cerebellar
source space for MNE-Python (:func:`setup_full_source_space`).
:func:`get_cerebellum_data` is a one-time setup step that downloads the atlas and
the nnU-Net models used by the first two stages.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_cerebellum_data
   segment_cerebellum
   create_cerebellar_surface
   setup_full_source_space


Visualization
-------------

.. currentmodule:: cmb.visualization

:func:`plot_normal` and :func:`plot_inflated` require the ``viz`` extra (PyVista); the
non-blocking Qt backend (``backend="pyvistaqt"``) also needs the ``viz-qt`` extra
(see the installation section on the :doc:`frontpage <index>`). :func:`plot_flatmap` and :func:`plot_sagittal` only
need matplotlib.

:func:`get_plot_data_from_stc` does not plot anything itself: it converts an MNE
source estimate into the cortex and cerebellum data arrays that the plotting
functions take.

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

   join_cortical_source_spaces
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

