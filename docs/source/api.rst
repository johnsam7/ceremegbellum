API reference
=============

The package is under active development, and API changes are possible.

Pipeline
--------

.. currentmodule:: cmb

The main workflow has three stages: segment the subject's cerebellum
(:func:`segment_cerebellum`), fit the atlas surface to it
(:func:`create_cerebellar_surface`), and build a combined cerebral and cerebellar
source space for MNE-Python.
:func:`get_cerebellum_data` is a one-time setup step that downloads the atlas and
the nnU-Net models used by the first two stages.

The third stage can be done in two ways, depending on the MNE-Python version:

- With current MNE-Python releases, :func:`setup_full_source_space` returns
  ``[cortex, cerebellum]``, with the two cortical hemispheres joined into one
  source space.
- The next MNE-Python release adds ``mne.setup_subcortical_source_space``, which
  builds the cerebellar source space directly from the mesh written by
  :func:`create_cerebellar_surface` (pass ``keep_largest_component=False`` to keep
  the vertex indices aligned with the atlas). Adding it to
  :func:`mne.setup_source_space` gives a mixed source space
  ``[lh, rh, cerebellum]``, and the inverse solution is a
  :class:`mne.MixedSourceEstimate`.

Both give the same source points and forward solution. In either layout the
cerebellum is the last source space, ``fwd["src"][-1]``.

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

The plotting functions expect the cortex as a single source space. With a mixed
``[lh, rh, cerebellum]`` source space, use
``join_cortical_source_spaces(fwd["src"][:2])`` to get one.
:func:`~cmb.visualization.get_plot_data_from_stc` does this automatically.

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

