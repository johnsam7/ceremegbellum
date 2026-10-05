"""Visualization functions for cerebellar cortical data.

Provides plotting in normal 3D and inflated 3D views using PyVista, as well as flatmap
view using Matplotlib.
"""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
#          Teemu Taivainen
# Created: November, 2021 (Modified: July, 2026)
# License: MIT
# ---------------------------------------------------------------------------

import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING, Literal, cast

# Use non-interactive backend when no display is available
import matplotlib

if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")
    _OFFSCREEN = True
else:
    _OFFSCREEN = False

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mne
import nibabel as nib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mne.morph import _hemi_morph
from numpy.typing import NDArray

from .source_space import _join_source_spaces

# This block is ONLY read by linters and type checkers (like mypy, Pylance)
# At runtime, it evaluates to False, keeping PyVista optional.
if TYPE_CHECKING:
    import pyvista as pv

logger = logging.getLogger(__name__)


def plot_cerebellum_data(*args, **kwargs) -> None:
    """Plot cerebellum data (DEPRECATED).

    This function is retained as a compatibility shim and no longer performs any
    plotting.
    """
    del args, kwargs

    warnings.warn(
        "plot_cerebellum_data is deprecated and does nothing. "
        "Please prepare your data for plotting using morph_cerebellum_data "
        "or morph_cortex_data, and then use plot_normal, plot_inflated, or "
        "plot_flatmap for visualization.",
        DeprecationWarning,
        stacklevel=2,
    )


def _interpolate_cerebellum_data(
    data: NDArray[np.floating],
    data_indices: NDArray[np.intp],
    subsampling: Literal["dense", "sparse"],
    cerebellum_geo: dict,
) -> NDArray[np.float64]:
    """Interpolate cerebellar data to cerebellar mesh with specified subsampling.

    Vertices on the (subsampled) cerebellar surface are filled in via
    iterative nearest-neighbor averaging.

    Parameters
    ----------
    data : ndarray of float, shape (n_known,)
        Values at known vertices in the cerebellar source space.
    data_indices : ndarray of int, shape (n_known,)
        Indices of the known vertices in the specified subsampling of the cerebellar
        surface mesh.
    subsampling : "dense" | "sparse"
        Subsampling of the cerebellar surface mesh corresponding to `data_indices`.
    cerebellum_geo : dict
        Cerebellum geometry object.

    Returns
    -------
    ndarray of float, shape (n_vertices,)
        Data values across the full cerebellar surface mesh corresponding
        to the specified subsampling.
    """
    if (
        data.ndim != 1
        or data_indices.ndim != 1
        or data.shape[0] != data_indices.shape[0]
    ):
        raise ValueError("data and data_indices must be 1D arrays of the same length.")
    if subsampling not in ["dense", "sparse"]:
        raise ValueError("subsampling must be either 'dense' or 'sparse'.")

    # Get number of vertices in specified subsampling of cerebellum.
    n_verts = cerebellum_geo["dw_data"][subsampling + "_verts"].shape[0]

    # Initialized interpolated data with NaNs, and fill in the known values.
    data_interpolated = np.full(n_verts, np.nan, dtype=np.float64)
    data_interpolated[data_indices] = data

    logger.info(
        f"Interpolating cerebellar data at {len(data_indices)} known vertices to "
        f"{n_verts} vertices in the {subsampling} subsampling of the cerebellar "
        f"surface mesh."
    )

    # Iteratively fill in NaN values by averaging over neighboring vertices until
    # no NaN values remain.
    nan_verts = np.where(np.isnan(data_interpolated))[0]
    iteration = 0
    while len(nan_verts) > 0:
        iteration += 1
        logger.debug(
            f"Interpolation iteration {iteration}: {len(nan_verts)} NaN vertices "
            "remaining."
        )
        # Get list of NumPy arrays, where each array contains the indices of the
        # neighboring vertices for a given vertex that has a NaN value.
        vert_neighbors = [
            cerebellum_geo["dw_data"][subsampling + "_vert_to_neighbor"][ind]
            for ind in nan_verts
        ]
        # Catch warnings for mean of empty slice, which is intended behavior when a
        # vertex has no neighbors with non-NaN values.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            neighbor_means = np.array(
                [
                    np.nanmean(data_interpolated[vert_neighbor_group])
                    for vert_neighbor_group in vert_neighbors
                ],
                dtype=np.float64,
            )
        resolved = ~np.isnan(neighbor_means)
        if not np.any(resolved):
            msg = (
                "Cerebellum interpolation stalled: some vertices could not be "
                "interpolated because they have no neighbors with known values. "
                "Leaving unresolved vertices as NaN."
            )
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            break

        data_interpolated[nan_verts[resolved]] = neighbor_means[resolved]
        nan_verts = np.where(np.isnan(data_interpolated))[0]

    logger.info("Cerebellum interpolation complete")

    return data_interpolated


def morph_cortex_data(
    cort_data: NDArray[np.floating],
    fwd_cortex_src: dict,
    smooth: int | None | Literal["nearest"] = None,
) -> NDArray[np.floating]:
    """Interpolate and optionally smooth cortex data to dense surface vertices.

    Morphs from the vertices used (in forward solution) to the full cortical mesh.

    Parameters
    ----------
    cort_data : ndarray of float, shape (n_used,)
        Data to be morphed to the full cortical mesh. The length must
        equal the number of **used** vertices in the cortical source space
        (i.e., `fwd_cortex_src['nuse']`).
    fwd_cortex_src : dict
        Cortical source space dictionary used in the forward solution,
        typically `fwd['src'][0]`.
    smooth : int | "nearest" | None
        Passed for `mne.morph._hemi_morph`.
        Controls spatial interpolation smoothing. If an integer, applies exactly
        that many iterative averaging steps (0 leaves data at sparse vertices only).
        If ``None`` (default), automatically iterates until all unmapped vertices are
        filled (capped at 100 steps). If ``"nearest"``, maps every vertex to the single
        closest source vertex without blending.

    Returns
    -------
    ndarray of float, shape (n_vertices,)
        Morphed cortical data on the full cortical mesh.
    """
    if cort_data.shape != (len(fwd_cortex_src["vertno"]),):
        raise ValueError(
            "cort_data must be a 1D array with length equal to the number of used "
            f"vertices in the cortical source space ({len(fwd_cortex_src['vertno'])}), "
            f"but got shape {cort_data.shape}."
        )
    logger.info(
        f"Morphing cortical data from {cort_data.shape[0]} vertices to full "
        f"cortical mesh with {fwd_cortex_src['np']} vertices using {smooth} "
        "smoothing steps."
    )
    morph = _hemi_morph(
        fwd_cortex_src["tris"],
        np.arange(fwd_cortex_src["np"]),
        fwd_cortex_src["vertno"],
        smooth,
        maps=None,
        warn=True,
    )
    result = morph @ cort_data[:, None]

    return result.ravel()  # Return as 1D array


def plot_normal(
    src_cerebellum: dict,
    cerebellum_data: NDArray[np.floating],
    src_cortex: dict | None = None,
    cortex_data: NDArray[np.floating] | None = None,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    show: bool = True,
    notebook_inline: bool = False,
    offscreen: bool | None = None,
    screenshot_fname: str | None = None,
    backend: Literal["pyvista", "pyvistaqt"] = "pyvista",
) -> "pv.BasePlotter":
    """Plot cerebellum and cortex in normal 3D view using PyVista.

    Parameters
    ----------
    src_cerebellum : dict
        Cerebellar source space dictionary, typically `fwd['src'][1]`.
    cerebellum_data : ndarray of float, shape (n_vertices,)
        Data to be visualized on the cerebellum, where n_vertices is the number of
        vertices in the dense triangulation of the cerebellar source space, i.e.
        `len(src_cerebellum['rr'])`.
    src_cortex : dict | None
        Cortical source space dictionary, typically `fwd['src'][0]`.
        If None (default), only the cerebellum will be plotted.
    cortex_data : ndarray of float, shape (n_vertices,) | None
        Data to be visualized on the cortex, where n_vertices is the number of
        vertices in the dense triangulation of the cortical source space, i.e.
        `len(src_cortex['rr'])`. If None (default), will plot zeros for the cortex when
        `src_cortex` is provided.
    cmap : str | None
        Color map to pass for PyVista plotter. If None (default), the PyVista default
        color map is used.
    clim : tuple of float | None
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the data across both cerebellum and cortex.
    show : bool
        Whether to show the plot immediately, by default True.
    notebook_inline : bool
        Whether to render the plot inline in a Jupyter notebook, by default False.
    offscreen : bool | None
        Whether to render the plot offscreen, by default None, which means the behavior
        will be determined by DISPLAY environment variable and OS type.
    screenshot_fname : str | None
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.
    backend : "pyvista" | "pyvistaqt"
        Plotting backend, by default "pyvista". "pyvista" uses a plain
        ``pyvista.Plotter`` whose ``show()`` blocks until the window is closed.
        "pyvistaqt" uses a non-blocking ``pyvistaqt.BackgroundPlotter`` that opens a
        Qt window and keeps it interactive (requires ``pyvistaqt`` and a Qt binding,
        installed by the ``viz-qt`` extra, and a display). With "pyvistaqt", the
        window closes when a plain Python script exits; use it from IPython
        (``%gui qt``) or call ``plotter.app.exec_()`` to keep it open. On Linux,
        ``QT_QPA_PLATFORM`` defaults to ``"xcb"`` (if not already set) so that it
        works on Wayland sessions; this has no effect if a Qt application was
        already created.

    Returns
    -------
    pv.BasePlotter
        The plotter object (``pyvista.Plotter`` or ``pyvistaqt.BackgroundPlotter``).
        The cerebellum mesh can be accessed and manipulated via
        ``plotter.actors['cerebellum_mesh']`` and the cortex mesh (if provided) via
        ``plotter.actors['cortex_mesh']``.
    """
    if len(cerebellum_data) != len(src_cerebellum["rr"]):
        raise ValueError(
            "cerebellum_data must have the same number of elements as the number of "
            "vertices in the dense triangulation of the cerebellar source space. "
            f"Expected {len(src_cerebellum['rr'])}, got {len(cerebellum_data)}."
        )
    if cortex_data is not None:
        if src_cortex is None:
            raise ValueError("src_cortex must be provided if cortex_data is provided.")
        if len(cortex_data) != len(src_cortex["rr"]):
            raise ValueError(
                "cortex_data must have the same number of elements as the number of "
                "vertices in the dense triangulation of the cortical source space. "
                f"Expected {len(src_cortex['rr'])}, got {len(cortex_data)}."
            )
    if clim is None:
        clim = _determine_global_clim(cerebellum_data, cortex_data)

    plotter, offscreen = _make_plotter(backend, offscreen, notebook_inline, show)

    cerebellum_mesh = _make_pyvista_mesh(src_cerebellum, cerebellum_data)
    plotter.add_mesh(
        cerebellum_mesh,
        scalars="scalars",
        cmap=cmap,
        scalar_bar_args={"color": "black"},
        clim=clim,
        name="cerebellum_mesh",
    )

    if src_cortex is not None:
        if cortex_data is None:
            # If no cortex data is provided, just plot zeros for the cortex.
            cortex_data = np.zeros(len(src_cortex["rr"]))

        cortex_mesh = _make_pyvista_mesh(src_cortex, cortex_data)
        plotter.add_mesh(
            cortex_mesh, scalars="scalars", cmap=cmap, clim=clim, name="cortex_mesh"
        )

    _finalize_plotter(
        plotter,
        focal_point=cerebellum_mesh.center,
        backend=backend,
        show=show,
        offscreen=offscreen,
        screenshot_fname=screenshot_fname,
        view_name="normal",
    )

    return plotter


def _import_pyvista():
    """Import PyVista lazily with a helpful error message."""
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyVista is required for 3D plotting. Please install it via "
            "'pip install pyvista'."
        ) from None
    return pv


def _make_plotter(
    backend: Literal["pyvista", "pyvistaqt"],
    offscreen: bool | None,
    notebook_inline: bool,
    show: bool,
) -> tuple["pv.BasePlotter", bool]:
    """Create a plotter for the requested backend.

    Parameters
    ----------
    backend : "pyvista" | "pyvistaqt"
        Plotting backend.
    offscreen : bool | None
        Whether to render offscreen. If None, determined from the environment for the
        "pyvista" backend.
    notebook_inline : bool
        Whether to render inline in a Jupyter notebook ("pyvista" backend only).
    show : bool
        Whether the "pyvistaqt" window is shown on creation.

    Returns
    -------
    plotter : pv.BasePlotter
        The created plotter with a white background.
    offscreen : bool
        The resolved offscreen setting.
    """
    if backend == "pyvista":
        pv = _import_pyvista()
        if offscreen is None:
            # Determine based on environment variable and OS type.
            offscreen = _OFFSCREEN
        plotter = pv.Plotter(
            window_size=[1200, 1200], off_screen=offscreen, notebook=notebook_inline
        )
    elif backend == "pyvistaqt":
        if notebook_inline:
            raise ValueError(
                "notebook_inline=True is not supported with backend='pyvistaqt'. "
                "Use backend='pyvista' instead."
            )
        if offscreen:
            raise ValueError(
                "offscreen rendering is not supported with backend='pyvistaqt'. "
                "Use backend='pyvista' instead."
            )
        if _OFFSCREEN:
            raise RuntimeError(
                "backend='pyvistaqt' requires a display, but none was detected "
                "(DISPLAY is not set). Use backend='pyvista' instead."
            )
        _import_pyvista()
        if sys.platform.startswith("linux"):
            # VTK renders through X11, so on Wayland sessions Qt must use the xcb
            # (XWayland) platform too, otherwise the render window fails with
            # "BadWindow". Respect an explicit user setting.
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        try:
            import pyvistaqt as pvqt
        except ImportError as err:
            raise ImportError(
                "backend='pyvistaqt' requires pyvistaqt and a Qt binding. Please "
                "install them via 'pip install \"cmb[viz-qt]\"' (or e.g. "
                "'pip install pyvistaqt pyqt6' to use another binding). "
                f"Original error: {err}"
            ) from None
        offscreen = False
        # BackgroundPlotter subclasses BasePlotter at runtime, but type checkers
        # cannot see it through qtpy's dynamically resolved Qt base classes.
        plotter = cast(
            "pv.BasePlotter",
            pvqt.BackgroundPlotter(window_size=(1200, 1200), show=show),
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Expected 'pyvista' or 'pyvistaqt'."
        )

    plotter.set_background(color="white")  # pyright: ignore[reportCallIssue]
    return plotter, offscreen


def _finalize_plotter(
    plotter: "pv.BasePlotter",
    focal_point: tuple[float, float, float],
    backend: Literal["pyvista", "pyvistaqt"],
    show: bool,
    offscreen: bool,
    screenshot_fname: str | None,
    view_name: str,
) -> None:
    """Set the camera, save an optional screenshot and show the plot."""
    plotter.camera.position = (0, -1, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.focal_point = focal_point
    plotter.reset_camera()  # pyright: ignore[reportCallIssue]

    if screenshot_fname is not None:
        if backend == "pyvistaqt":
            # Make sure the Qt render window is up to date before grabbing it.
            plotter.render()
        plotter.screenshot(screenshot_fname)
        logger.info(f"Saved {view_name} view to {screenshot_fname}")

    if show and offscreen:
        warnings.warn(
            "Showing the plot is not supported in offscreen mode.",
            UserWarning,
            stacklevel=3,
        )
    elif show and backend == "pyvista":
        # BackgroundPlotter is shown on creation and does not block.
        plotter.show()


def _determine_global_clim(
    cerebellum_data: NDArray[np.floating],
    cortex_data: NDArray[np.floating] | None,
) -> tuple[float, float]:
    """Determine color limits for plotting.

    Sets the color limits to the min and max of the data across both cerebellum
    and cortex.
    """
    cerebellum_min = np.nanmin(cerebellum_data)
    cerebellum_max = np.nanmax(cerebellum_data)
    cortex_min = np.nanmin(cortex_data) if cortex_data is not None else cerebellum_min
    cortex_max = np.nanmax(cortex_data) if cortex_data is not None else cerebellum_max

    clim = (
        float(min(cerebellum_min, cortex_min)),
        float(max(cerebellum_max, cortex_max)),
    )

    return clim


def _make_pyvista_mesh(src_space: dict, data: NDArray[np.floating]) -> "pv.PolyData":
    """Create a PyVista PolyData object for the visualization.

    Makes a triangular mesh with a scalar value in each vertex using all vertices in
    the given source space.

    Parameters
    ----------
    src_space : dict
        The source space dictionary, typically `fwd['src'][0]` for cortex or
        `fwd['src'][1]` for cerebellum.
    data : ndarray of float, shape (n_vertices,)
        Data to be visualized on the source space surface, where n_vertices is the
        number of vertices in the source space, i.e. `len(src_space['rr'])`.
    """
    pv = _import_pyvista()
    verts = src_space["rr"]
    faces = src_space["tris"]

    # Add a column of 3s to tell PyVista that these are triangles (3 vertices per face).
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])
    mesh = pv.PolyData(verts, pv_faces)
    mesh.point_data["scalars"] = data

    return mesh


def plot_inflated(
    cerebellum_geo: dict,
    cerebellum_data: NDArray[np.floating],
    subsampling: Literal["dense", "sparse"],
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    show: bool = True,
    notebook_inline: bool = False,
    offscreen: bool | None = None,
    screenshot_fname: str | None = None,
    backend: Literal["pyvista", "pyvistaqt"] = "pyvista",
) -> "pv.BasePlotter":
    """Plot cerebellum in inflated 3D view using PyVista.

    Parameters
    ----------
    cerebellum_geo : dict
        Cerebellum geometry object.
    cerebellum_data : ndarray of float, shape (n_vertices,)
        Data to be visualized on the cerebellum, where n_vertices is the number of
        vertices in the specified subsampling of the cerebellar surface mesh.
    subsampling : "dense" | "sparse"
        Subsampling of the cerebellar surface mesh corresponding to `cerebellum_data`.
    cmap : str | None
        Color map to pass for PyVista plotter. If None (default), the PyVista default
        color map is used.
    clim : tuple of float | None
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the `cerebellum_data`.
    show : bool
        Whether to show the plot immediately, by default True.
    notebook_inline : bool
        Whether to render the plot inline in a Jupyter notebook, by default False.
    offscreen : bool | None
        Whether to render the plot offscreen, by default None, which means the behavior
        will be determined by DISPLAY environment variable and OS type.
    screenshot_fname : str | None
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.
    backend : "pyvista" | "pyvistaqt"
        Plotting backend, by default "pyvista". See :func:`plot_normal` for details.

    Returns
    -------
    pv.BasePlotter
        The plotter object (``pyvista.Plotter`` or ``pyvistaqt.BackgroundPlotter``).
        The cerebellum mesh can be accessed and manipulated via
        ``plotter.actors['cerebellum_mesh']``.
    """
    pv = _import_pyvista()

    if clim is None:
        clim = (float(np.nanmin(cerebellum_data)), float(np.nanmax(cerebellum_data)))
    # Get indices of the vertices used in specified subsampling.
    vertex_indices = cerebellum_geo["dw_data"][subsampling]
    inflated_verts_subsampled = cerebellum_geo["verts_inflated_fs"][vertex_indices]

    if len(cerebellum_data) != len(inflated_verts_subsampled):
        raise ValueError(
            "cerebellum_data must have the same number of elements as the number of "
            f"vertices in the {subsampling} subsampling of cerebellum. "
            f"Expected {len(inflated_verts_subsampled)}, got {len(cerebellum_data)}."
        )

    faces = cerebellum_geo["dw_data"][subsampling + "_tris"]
    # Add a column of 3s to tell PyVista that these are triangles (3 vertices per face).
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])

    cerebellum_mesh = pv.PolyData(inflated_verts_subsampled, pv_faces)
    cerebellum_mesh.point_data["scalars"] = cerebellum_data

    plotter, offscreen = _make_plotter(backend, offscreen, notebook_inline, show)
    plotter.add_mesh(
        cerebellum_mesh,
        scalars="scalars",
        cmap=cmap,
        clim=clim,
        scalar_bar_args={"color": "black"},
        name="cerebellum_mesh",
    )

    _finalize_plotter(
        plotter,
        focal_point=cerebellum_mesh.center,
        backend=backend,
        show=show,
        offscreen=offscreen,
        screenshot_fname=screenshot_fname,
        view_name="inflated",
    )

    return plotter


def plot_flatmap(
    cerebellum_geo: dict,
    cerebellum_data: NDArray[np.floating],
    subsampling: Literal["dense", "sparse"],
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    show: bool = True,
    offscreen: bool | None = None,
    screenshot_fname: str | None = None,
) -> Figure:
    """Plot cerebellum in flatmap view using Matplotlib.

    Parameters
    ----------
    cerebellum_geo : dict
        Cerebellum geometry object.
    cerebellum_data : ndarray of float, shape (n_vertices,)
        Data to be visualized on the cerebellum, where n_vertices is the number of
        vertices in the specified subsampling of the cerebellar surface mesh.
    subsampling : "dense" | "sparse"
        Subsampling of the cerebellar surface mesh corresponding to `cerebellum_data`.
    cmap : str | None
        Color map to pass for Matplotlib. If None (default), will use "bwr" if the data
        crosses zero, and "Reds" otherwise.
    clim : tuple of float | None
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the `cerebellum_data`.
    show : bool
        Whether to show the plot immediately, by default True.
    offscreen : bool | None
        Whether to render the plot offscreen, by default None, which means the behavior
        will be determined by DISPLAY environment variable and OS type.
    screenshot_fname : str | None
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.

    Returns
    -------
    Figure
        The Matplotlib figure object.
    """
    if offscreen is None:
        # Determine based on environment variable and OS type.
        offscreen = _OFFSCREEN
    norm, ticks, cmap = _get_flatmap_color_mapping(cerebellum_data, clim, cmap)

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Plot outer borders of flattened cerebellum.
    for flatmap in cerebellum_geo["flatmap_outlines"]:
        # Minus x-coord to keep in neurological coordinates
        # i.e. plot left on the left.
        ax.plot(-flatmap[:, 0], flatmap[:, 1], linestyle="--", linewidth=0.4, c="k")

    # Plot each region one by one.
    triconf = None
    for region_key in cerebellum_geo["flatmap_inds"]:
        triangulation, region_vertex_indices = _build_flatmap_region_triangulation(
            cerebellum_geo, region_key, subsampling
        )
        region_data = cerebellum_data[region_vertex_indices]

        triconf = ax.tripcolor(
            triangulation,
            region_data,
            cmap=cmap,
            norm=norm,
            shading="gouraud",  # smooth color transitions across triangles
        )

    assert triconf is not None, "Triangulation contour plot should be created."
    fig.colorbar(triconf, ax=ax, ticks=ticks)

    _draw_anatomical_annotations(ax)

    if screenshot_fname:
        fig.savefig(screenshot_fname, dpi=300, bbox_inches="tight", transparent=True)
        logger.info(f"Saved flatmap view to {screenshot_fname}")

    if show and offscreen:
        warnings.warn(
            "Showing the plot is not supported in offscreen mode.",
            UserWarning,
            stacklevel=2,
        )
    elif show:
        plt.show()

    return fig


def _draw_anatomical_annotations(ax: Axes):
    """Draws anatomical annotations on the cerebellum flatmap."""
    # Inner Borders
    borders = [
        np.array(
            [
                [-110, 918],
                [-86, 925],
                [-52, 935],
                [-16, 942],
                [23, 965],
                [57, 980],
                [78, 983],
                [123, 966],
            ]
        ),
        np.array([[-165, 257], [-126, 260], [-80, 300]]),
        np.array([[96, 313], [230, 148]]),
        np.array([[-239, -49], [-178, -117]]),
        np.array([[255, -211], [244, -119], [265, -75], [293, -71]]),
    ]
    for border in borders:
        ax.plot(-border[:, 0], border[:, 1], linestyle="--", linewidth=0.4, c="k")

    # Text Labels
    text_params = {"fontsize": 8, "verticalalignment": "top"}
    texts = [
        (-610, 1175, " Lobules I-V \n (anterior lobe)"),
        (-490, 840, "Lobule VI"),
        (-450, 441, "Crus I"),
        (-700, 100, " Crus II/\n Lobule VIIb"),
        (-740, -200, "Lobule VIII"),
        (-670, -590, " Lobule IX \n (tonsil)"),
        (-370, -670, " Lobule X \n (flocculus)"),
        (40, -710, "Inferior vermis"),
        (-380, 1480, "Left", {"fontweight": "bold"}),
        (200, 1480, "Right", {"fontweight": "bold"}),
    ]

    for item in texts:
        t_params = {**text_params, **item[3]} if len(item) == 4 else text_params
        ax.text(item[0], item[1], item[2], **t_params)

    # Arrows
    arrow_params = {
        "head_width": 20,
        "head_length": 20,
        "linewidth": 0.5,
        "fc": "k",
        "ec": "k",
    }
    arrows = [(-350, -580, 82, 64), (-230, -660, 80, 45), (20, -750, 0, 130)]
    for arr in arrows:
        ax.arrow(arr[0], arr[1], arr[2], arr[3], **arrow_params)


def _get_flatmap_color_mapping(
    data: NDArray[np.floating], clim: tuple[float, float] | None, cmap: str | None
) -> tuple[mcolors.TwoSlopeNorm | mcolors.Normalize, list[float], str]:
    """Determine color limits, normalization and colormap automatically."""
    if clim is None:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    else:
        vmin, vmax = clim

    # Determine if the map needs to be centered on zero
    crosses_zero = vmin < 0 and vmax > 0

    if cmap is None:
        cmap = "bwr" if crosses_zero else "Reds"

    if crosses_zero:
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        ticks = [vmin, 0, vmax]
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ticks = [vmin, vmax]

    return norm, ticks, cmap


def _build_flatmap_region_triangulation(
    cerebellum_geo: dict,
    region_key: str,
    subsampling: Literal["dense", "sparse"],
) -> tuple[mtri.Triangulation, NDArray[np.intp]]:
    """Build a triangulation for a specific region of the cerebellum flatmap.

    Parameters
    ----------
    cerebellum_geo : dict
        The cerebellum geometry object.
    region_key : str
        The key for the region of interest.
    subsampling : "dense" | "sparse"
        The chosen subsampling of the cerebellar surface mesh.

    Returns
    -------
    tuple of mtri.Triangulation and ndarray of int
        A tuple containing the triangulation for the specified region and the
        indices of the vertices belonging to that region within the subsampled mesh.
    """
    # Get indices of vertices that belong to this region
    # (from all the vertices in the full triangulation).
    region_vertex_global_indices = cerebellum_geo["flatmap_inds"][region_key]
    # Get indices of vertices considered with chosen subsampling.
    subsampled_vertex_global_indices = cerebellum_geo["dw_data"][subsampling]
    # Get mask that is True for those subsampled vertices that belong to the region.
    region_verts_mask = np.isin(
        subsampled_vertex_global_indices,
        region_vertex_global_indices,
    )
    # Extract both the global indices of the vertices within subsampled region
    # and the local indices of those vertices within the subsampled region.
    region_vertex_indices_global = subsampled_vertex_global_indices[region_verts_mask]
    region_vertex_indices = np.nonzero(region_verts_mask)[0]

    # Get coordinates of corresponding flat vertices.
    flat_verts = cerebellum_geo["verts_flatmap"][region_vertex_indices_global, :]

    # Get all triangles in the subsampled mesh and filter to those that are fully within
    # the region.
    subsampled_tris = cerebellum_geo["dw_data"][subsampling + "_tris"]
    region_tris_mask = (
        # See if each vertex belongs to the region.
        np.isin(subsampled_tris, region_vertex_indices)
        # Yield True if all the vertices of the triangle belong to the region.
        .all(axis=1)
    )
    # contains local indices of vertices in the subsampled mesh
    region_tris = subsampled_tris[region_tris_mask, :]

    # Move from indices withhin the subsampled mesh to indices within the region.
    tris_flat = np.searchsorted(region_vertex_indices, region_tris)

    triangulation = mtri.Triangulation(
        -flat_verts[:, 0], flat_verts[:, 1], tris_flat
    )  # minus x-coord to keep in neurological coordinates

    return triangulation, region_vertex_indices


def morph_cerebellum_data(
    data: NDArray[np.floating],
    fwd_cerebellum_src: dict,
    cerebellum_geo: dict,
    subsampling: Literal["dense", "sparse"],
    smoothing_steps: int = 0,
) -> NDArray[np.floating]:
    """Interpolate cerebellar data to specified subsampling and optionally smooth it.

    Vertices without data are filled in via iterative nearest-neighbor averaging.

    Parameters
    ----------
    data : ndarray of float, shape (n_used,)
        Data to be plotted on the cerebellum, where n_used is the number of vertices
        that are used in the forward solution for the cerebellar source space,
        i.e. `len(fwd_cerebellum_src['vertno'])`.
    fwd_cerebellum_src : dict
        The cerebellar source space used in the computation of the forward solution,
        typically `fwd['src'][1]`.
    cerebellum_geo : dict
        Cerebellum geometry object.
    subsampling : "dense" | "sparse"
        Subsampling of the cerebellar surface mesh to which the data should be
        interpolated. Must match the subsampling used for calculating
        the forward solution.
    smoothing_steps : int
        Number of smoothing iterations to apply to the interpolated data. Each
        iteration averages the value at each vertex with its neighbors. By default 0,
        which means no smoothing is applied.

    Returns
    -------
    ndarray of float, shape (n_vertices,)
        Data values across the full cerebellar surface mesh corresponding
        to the specified subsampling.
    """
    # Indices of the vertices in the cerebellar source space that are used in the
    # forward solution. Data is provided for these vertices.
    data_indices = fwd_cerebellum_src["vertno"]
    # Help IDE type checkers with assertion.
    assert isinstance(data_indices, np.ndarray), (
        "fwd_src[1]['vertno'] must be a numpy array."
    )
    if data_indices.shape[0] != data.shape[0]:
        raise ValueError(
            "data must have the same number of elements as the number of vertices "
            "used in the forward solution for the cerebellar source space, i.e. "
            "len(fwd_cerebellum_src['vertno'])."
        )
    data_interpolated = _interpolate_cerebellum_data(
        data,
        data_indices=data_indices,
        subsampling=subsampling,
        cerebellum_geo=cerebellum_geo,
    )
    if smoothing_steps <= 0:
        return data_interpolated

    # Apply smoothing to the interpolated data by averaging over neighboring vertices.
    vert_to_neighbors = cerebellum_geo["dw_data"][subsampling + "_vert_to_neighbor"]
    for step in range(smoothing_steps):
        logger.info(
            f"Applying smoothing {step + 1}/{smoothing_steps} to cerebellum data."
        )
        # Make copy to keep smoothing independent of vertex iteration order.
        new_values = data_interpolated.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            for vert in range(data_interpolated.shape[0]):
                new_values[vert] = np.nanmean(
                    data_interpolated[vert_to_neighbors[vert]]
                )
        data_interpolated = new_values  # update with smoothed values

    return data_interpolated


def get_plot_data_from_stc(
    stc: mne.SourceEstimate | mne.MixedSourceEstimate,
    fwd_src: mne.SourceSpaces,
    time_point: float,
    cerebellum_geo: dict,
    cerebellum_subsampling: Literal["sparse", "dense"],
    cerebellum_idx: int = 1,
    cortex_smooth: int | None | Literal["nearest"] = None,
    cerebellum_smooth: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get the data of one time point for plotting from a SourceEstimate object.

    This works as a bridge between the `SourceEstimate`object and the visualization
    functions in this package. Supports both legacy CMB source spaces where cerebellum
    source space is the second element in the `SourceSpaces` list, and mixed source
    spaces where cerebellum source space is the third element in the `SourceSpaces`
    list. This function will be hopefully removed in the future when MNE-Python
    handles the plotting natively.

    Parameters
    ----------
    stc : mne.SourceEstimate | mne.MixedSourceEstimate
        The source estimate object containing the data to plot.
    fwd_src : mne.SourceSpaces
        The source spaces used in the forward model.
    time_point : float
        The time point to extract data for plotting.
    cerebellum_geo : dict
        The cerebellum geometry data loaded from the cerebellum_geo file.
    cerebellum_subsampling : "sparse" | "dense"
        The subsampling used for the cerebellum source space.
    cerebellum_idx : int
        The index of the cerebellum source space in the `SourceSpaces` list.
        Allowed values are 1 (legacy CMB source space) or 2 (mixed source space).
        Defaults to 1.
    cortex_smooth : int | "nearest" | None
        Passed for `mne.morph._hemi_morph`.
        Controls spatial interpolation smoothing. If an integer, applies exactly
        that many iterative averaging steps (0 leaves data at sparse vertices only).
        If ``None`` (default), automatically iterates until all unmapped vertices are
        filled (capped at 100 steps). If ``"nearest"``, maps every vertex to the single
        closest source vertex without blending.
    cerebellum_smooth : int
        Number of smoothing iterations to apply to the interpolated data. Each
        iteration averages the value at each vertex with its neighbors. By default 0,
        which means no smoothing is applied.

    Returns
    -------
    cortex_data : ndarray of float, shape (n_vertices,)
        Data for each vertex in the full cortical mesh at the specified time point.
    cerebellum_data : ndarray of float, shape (n_vertices,)
        Data for each vertex in the subsampled cerebellar mesh at the specified time
        point.
    """
    if cerebellum_idx not in [1, 2]:
        raise ValueError(f"Invalid cerebellum index: {cerebellum_idx}. Must be 1 or 2.")
    if cerebellum_idx == 1:
        n_cortex_verts = fwd_src[0]["nuse"]
        fwd_cortex_src = fwd_src[0]
        fwd_cerebellum_src = fwd_src[1]
    else:
        n_cortex_verts = fwd_src[0]["nuse"] + fwd_src[1]["nuse"]
        # Concatenate the two cortical source spaces to be compatible
        # with the visualization functions.
        fwd_cortex_src = _join_source_spaces(fwd_src[:2])
        fwd_cerebellum_src = fwd_src[2]
    assert isinstance(fwd_cortex_src, dict)
    assert isinstance(fwd_cerebellum_src, dict)

    stc_data = stc.data
    assert isinstance(stc_data, np.ndarray)
    time_idx = stc.time_as_index(time_point)[0]

    cortex_data = stc_data[:n_cortex_verts, time_idx]
    cerebellum_data = stc_data[n_cortex_verts:, time_idx]

    cortex_data = morph_cortex_data(
        cort_data=cortex_data, fwd_cortex_src=fwd_cortex_src, smooth=cortex_smooth
    )
    cerebellum_data = morph_cerebellum_data(
        data=cerebellum_data,
        fwd_cerebellum_src=fwd_cerebellum_src,
        cerebellum_geo=cerebellum_geo,
        subsampling=cerebellum_subsampling,
        smoothing_steps=cerebellum_smooth,
    )

    return cortex_data, cerebellum_data


def plot_sagittal(
    vol_fname: os.PathLike[str] | str,
    sag_ind: list[int] | None = None,
    mesh_fname: os.PathLike[str] | str | None = None,
    show_only_midline: bool = False,
    cmap: str = "gray_r",
    linewidth: float = 1.0,
    title: str = "Subject MRI sagittal slices",
) -> Figure:
    """Plot sagittal slices of a 3D MRI volume with optional surface mesh overlay.

    Parameters
    ----------
    vol_fname : path-like
        Path to the MRI volume file (e.g. orig.mgz).
    sag_ind : list of int | None
        List of sagittal slice indices to plot. If None, will plot 6 evenly spaced
        slices across the volume (default). If `show_only_midline` is True, this
        parameter is ignored.
    mesh_fname : path-like | None
        Path to the surface mesh file. If None (default), no surface mesh will be
        overlaid on the MRI slices.
    show_only_midline : bool
        If True, will only plot the midline sagittal slice of the volume.
        Default is False.
    cmap : str
        Colormap to use for displaying the MRI slices. Default is "gray_r".
    linewidth : float
        Line width for the surface mesh overlay. Default is 1.0.
    title : str
        Title for the figure. Default is "Subject MRI sagittal slices".

    Returns
    -------
    Figure
        The Matplotlib figure object.
    """
    from nibabel.affines import apply_affine

    mri_vol = nib.load(vol_fname)
    vol = mri_vol.get_fdata()  # pyright: ignore[reportAttributeAccessIssue]

    if mesh_fname is not None:
        rr, tris = mne.read_surface(mesh_fname)  # pyright: ignore
        # Transform the surface mesh coordinates from surface RAS to MRI voxel space.
        vox2ras_tkr = mri_vol.header.get_vox2ras_tkr()  # pyright: ignore[reportAttributeAccessIssue]
        tkr2vox = np.linalg.inv(vox2ras_tkr)
        rr = apply_affine(tkr2vox, rr)

        # Help IDE type checkers with assertions.
        assert isinstance(rr, np.ndarray), (
            "Surface mesh coordinates must be a numpy array."
        )
        assert isinstance(tris, np.ndarray), (
            "Surface mesh triangles must be a numpy array."
        )
    else:
        rr, tris = None, None

    if show_only_midline:
        sag_ind = [vol.shape[0] // 2]
    elif sag_ind is None:
        x_width = vol.shape[0]
        sag_ind = list(
            np.linspace(int(x_width * 0.1), int(x_width * 0.9), 6).astype(int)
        )

    # Dynamically size the grid
    n_plots = len(sag_ind)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.suptitle(title)

    for c, slice_idx in enumerate(sag_ind):
        image = vol[slice_idx, :, :]
        plt.subplot(rows, cols, c + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(f"Slice {slice_idx}")

        if tris is not None and rr is not None:
            z_0 = slice_idx
            cart_ind = 0
            xy = [x for x in range(3) if not x == cart_ind]
            intersecting_tris = []
            for tri in tris:
                rr_0 = rr[tri[0], :]
                rr_1 = rr[tri[1], :]
                rr_2 = rr[tri[2], :]
                if (
                    np.array(
                        [
                            np.sign((rr_0[cart_ind] - z_0) * (rr_1[cart_ind] - z_0)),
                            np.sign((rr_0[cart_ind] - z_0) * (rr_2[cart_ind] - z_0)),
                            np.sign((rr_1[cart_ind] - z_0) * (rr_2[cart_ind] - z_0)),
                        ]
                    )
                    == -1
                ).any():
                    intersecting_tris.append(tri)
            intersecting_tris = np.array(intersecting_tris)
            for int_tri in intersecting_tris:
                rr_0 = rr[int_tri[0], :]
                rr_1 = rr[int_tri[1], :]
                rr_2 = rr[int_tri[2], :]
                t_0 = (z_0 - rr_0[cart_ind]) / (rr_1[cart_ind] - rr_0[cart_ind])
                t_1 = (z_0 - rr_0[cart_ind]) / (rr_2[cart_ind] - rr_0[cart_ind])
                t_2 = (z_0 - rr_1[cart_ind]) / (rr_2[cart_ind] - rr_1[cart_ind])
                xy_points = []
                if t_0 > 0 and t_0 < 1:
                    xy_points.append(t_0 * rr_1[xy] + (1 - t_0) * rr_0[xy])
                if t_1 > 0 and t_1 < 1:
                    xy_points.append(t_1 * rr_2[xy] + (1 - t_1) * rr_0[xy])
                if t_2 > 0 and t_2 < 1:
                    xy_points.append(t_2 * rr_2[xy] + (1 - t_2) * rr_1[xy])
                xy_points = np.array(xy_points)
                plt.plot(
                    xy_points[:, 1], xy_points[:, 0], color="red", linewidth=linewidth
                )

    return fig
