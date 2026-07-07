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
import warnings
from typing import TYPE_CHECKING, Literal

# Use non-interactive backend when no display is available
import matplotlib

if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mne.morph import _hemi_morph

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


def interpolate_cerebellum_data(
    data: npt.NDArray[np.floating],
    data_indices: npt.NDArray[np.intp],
    subsampling: Literal["dense", "sparse"],
    cerebellum_geo: dict,
) -> npt.NDArray[np.float64]:
    """Interpolate cerebellar data to cerebellar mesh with specified subsampling.

    Vertices on the (subsampled) cerebellar surface are filled in via
    iterative nearest-neighbor averaging.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Values at known vertices in the cerebellar source space.
    data_indices : npt.NDArray[np.intp]
        Indices of the known vertices in the specified subsampling of the cerebellar
        surface mesh.
    subsampling : Literal["dense", "sparse"]
        Subsampling of the cerebellar surface mesh corresponding to `data_indices`.
    cerebellum_geo : dict
        Cerebellum geometry object.


    Returns
    -------
    npt.NDArray[np.float64]
        1D array of data values across the full cerebellar surface mesh corresponding
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
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            break

        data_interpolated[nan_verts[resolved]] = neighbor_means[resolved]
        nan_verts = np.where(np.isnan(data_interpolated))[0]

    logger.info("Cerebellum interpolation complete")

    return data_interpolated


def morph_cortex_data(
    cort_data: npt.NDArray[np.floating],
    fwd_cortex_src: dict,
    smooth: int | None | Literal["nearest"] = None,
) -> npt.NDArray[np.floating]:
    """Interpolate and optionally smooth cortex data to dense surface vertices.

    Morphs from the vertices used (in forward solution) to the full cortical mesh.

    Parameters
    ----------
    cort_data : npt.NDArray[np.floating]
        Data to be morphed to the full cortical mesh. Should be an 1D array with length
        equal to the number of **used** vertices in the cortical source space
        (i.e., `fwd_cortex_src['nuse']`).
    fwd_cortex_src : dict
        Cortical source space dictionary used in the forward solution,
        typically `fwd['src'][0]`.
    smooth : int | None | Literal["nearest"]
        Passed for `mne.morph._hemi_morph`.
        Controls spatial interpolation smoothing. If an integer, applies exactly
        that many iterative averaging steps (0 leaves data at sparse vertices only).
        If ``None``, automatically iterates until all unmapped vertices are filled
        (capped at 100 steps). If ``"nearest"``, maps every vertex to the single
        closest source vertex without blending.

    Returns
    -------
    npt.NDArray[np.floating]
        1D array of morphed cortical data on the full cortical mesh.
    """
    if cort_data.ndim != 1 or cort_data.shape[0] != len(fwd_cortex_src["vertno"]):
        raise ValueError(
            "cort_data must be a 1D array with length equal to the number of used "
            "vertices in the cortical source space (i.e., "
            "len(cortex_src['vertno']))."
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
    cerebellum_data: npt.NDArray[np.floating],
    src_cortex: dict | None = None,
    cortex_data: npt.NDArray[np.floating] | None = None,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    offscreen: bool = False,
    screenshot_fname: str | None = None,
) -> "pv.Plotter":
    """Plot cerebellum and cortex in normal 3D view using PyVista.

    Parameters
    ----------
    src_cerebellum : dict
        Cerebellar source space dictionary, typically `fwd['src'][1]`.
    cerebellum_data : npt.NDArray[np.floating]
        Data to be visualized on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the dense triangulation of
        the cerebellar source space, i.e. `len(src_cerebellum['rr'])`.
    src_cortex : dict | None, optional
        Cortical source space dictionary, typically `fwd['src'][0]`.
        If None (default), only the cerebellum will be plotted.
    cortex_data : npt.NDArray[np.floating] | None, optional
        Data to be visualized on the cortex. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the dense triangulation of the
        cortical source space, i.e. `len(src_cortex['rr'])`. If None (default), will
        plot zeros for the cortex when `src_cortex` is provided.
    cmap : str | None, optional
        Color map to pass for PyVista plotter.
    clim : tuple[float, float] | None, optional
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the data across both cerebellum and cortex.
    screenshot_fname : str | None, optional
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.

    Returns
    -------
    pv.Plotter
        The PyVista plotter object.
    """
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyVista is required for 3D plotting. Please install it via "
            "'pip install pyvista'."
        ) from None

    if len(cerebellum_data) != len(src_cerebellum["rr"]):
        raise ValueError(
            "cerebellum_data must have the same number of elements as the number of "
            "vertices in the dense triangulation of the cerebellar source space. "
            f"Expected {len(src_cerebellum['rr'])}, got {len(cerebellum_data)}."
        )
    if cortex_data is not None and src_cortex is None:
        raise ValueError("src_cortex must be provided if cortex_data is provided.")
    if clim is None:
        clim = _determine_global_clim(cerebellum_data, cortex_data)

    plotter = pv.Plotter(window_size=[1200, 1200], off_screen=offscreen)
    # Ignoring warning because my pyright is confused.
    plotter.set_background(color="white")  # pyright: ignore[reportCallIssue]

    cerebellum_mesh = _make_pyvista_mesh(src_cerebellum, cerebellum_data)
    plotter.add_mesh(
        cerebellum_mesh,
        scalars="scalars",
        cmap=cmap,
        scalar_bar_args={"color": "black"},
        clim=clim,
    )

    if src_cortex is not None:
        if cortex_data is None:
            # If no cortex data is provided, just plot zeros for the cortex.
            cortex_data = np.zeros(len(src_cortex["rr"]))

        cortex_mesh = _make_pyvista_mesh(src_cortex, cortex_data)
        plotter.add_mesh(cortex_mesh, scalars="scalars", cmap=cmap, clim=clim)

    plotter.camera.position = (0, -1, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.focal_point = cerebellum_mesh.center
    plotter.reset_camera()  # pyright: ignore[reportCallIssue]

    if screenshot_fname is not None:
        plotter.screenshot(screenshot_fname)
        logger.info(f"Saved normal view to {screenshot_fname}")

    if not offscreen:
        plotter.show()

    return plotter


def _determine_global_clim(
    cerebellum_data: npt.NDArray[np.floating],
    cortex_data: npt.NDArray[np.floating] | None,
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


def _make_pyvista_mesh(
    src_space: dict, data: npt.NDArray[np.floating]
) -> "pv.PolyData":
    """Create a PyVista PolyData object for the visualization.

    Makes a triangular mesh with a scalar value in each vertex using all vertices in
    the given source space.

    Parameters
    ----------
    src_space : dict
        The source space dictionary, typically `fwd['src'][0]` for cortex or
        `fwd['src'][1]` for cerebellum.
    data : npt.NDArray[np.floating]
        Data to be visualized on the source space surface. Should have shape
        (n_vertices,), where n_vertices is the number of vertices in the source space,
        i.e. `len(src_space['rr'])`.
    """
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyVista is required for 3D plotting. Please install it via "
            "'pip install pyvista'."
        ) from None
    verts = src_space["rr"]
    faces = src_space["tris"]

    # Add a column of 3s to tell PyVista that these are triangles (3 vertices per face).
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])
    mesh = pv.PolyData(verts, pv_faces)
    mesh.point_data["scalars"] = data

    return mesh


def plot_inflated(
    cerebellum_geo: dict,
    cerebellum_data: npt.NDArray[np.floating],
    subsampling: Literal["dense", "sparse"],
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    offscreen: bool = False,
    screenshot_fname: str | None = None,
) -> "pv.Plotter":
    """Plot cerebellum in inflated 3D view using PyVista.

    Parameters
    ----------
    cerebellum_geo : dict
        Cerebellum geometry object.
    cerebellum_data : npt.NDArray[np.floating]
        Data to be visualized on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the specified subsampling of
        the cerebellar surface mesh.
    subsampling : Literal["dense", "sparse"]
        Subsampling of the cerebellar surface mesh corresponding to `cerebellum_data`.
    cmap : str | None, optional
        Color map to pass for PyVista plotter.
    clim : tuple[float, float] | None, optional
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the `cerebellum_data`.
    offscreen : bool, optional
        Whether to render the plot offscreen, by default False.
    screenshot_fname : str | None, optional
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.

    Returns
    -------
    pv.Plotter
        The PyVista plotter object.
    """
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyVista is required for 3D plotting. Please install it via "
            "'pip install pyvista'."
        ) from None

    if clim is None:
        clim = (float(np.nanmin(cerebellum_data)), float(np.nanmax(cerebellum_data)))
    # Get indices of the vertices used in specified subsampling.
    vertex_indices = cerebellum_geo["dw_data"][subsampling]
    inflated_verts_subsampled = cerebellum_geo["verts_inflated_fs"][vertex_indices]

    faces = cerebellum_geo["dw_data"][subsampling + "_tris"]
    # Add a column of 3s to tell PyVista that these are triangles (3 vertices per face).
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])

    cerebellum_mesh = pv.PolyData(inflated_verts_subsampled, pv_faces)
    cerebellum_mesh.point_data["scalars"] = cerebellum_data

    plotter = pv.Plotter(window_size=[1200, 1200], off_screen=offscreen)
    plotter.set_background(color="white")  # pyright: ignore[reportCallIssue]
    plotter.add_mesh(
        cerebellum_mesh,
        scalars="scalars",
        cmap=cmap,
        clim=clim,
        scalar_bar_args={"color": "black"},
    )
    plotter.camera.position = (0, -1, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.focal_point = cerebellum_mesh.center
    plotter.reset_camera()  # pyright: ignore[reportCallIssue]

    if screenshot_fname is not None:
        plotter.screenshot(screenshot_fname)
        logger.info(f"Saved inflated view to {screenshot_fname}")

    if not offscreen:
        plotter.show()

    return plotter


def plot_flatmap(
    cerebellum_geo: dict,
    cerebellum_data: npt.NDArray[np.floating],
    subsampling: Literal["dense", "sparse"],
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    offscreen: bool = False,
    screenshot_fname: str | None = None,
) -> Figure:
    """Plot cerebellum in flatmap view using Matplotlib.

    Parameters
    ----------
    cerebellum_geo : dict
        Cerebellum geometry object.
    cerebellum_data : npt.NDArray[np.floating]
        Data to be visualized on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the specified subsampling of
        the cerebellar surface mesh.
    subsampling : Literal["dense", "sparse"]
        Subsampling of the cerebellar surface mesh corresponding to `cerebellum_data`.
    cmap : str | None, optional
        Color map to pass for Matplotlib. If None (default), will use "bwr" if the data
        crosses zero, and "Reds" otherwise.
    clim : tuple[float, float] | None, optional
        Color bar limits, by default None, which means that the clim will be
        set to the min and max of the `cerebellum_data`.
    offscreen : bool, optional
        Whether to render the plot offscreen, by default False.
    screenshot_fname : str | None, optional
        Filename to save the screenshot, by default None, which means no screenshot is
        saved.

    Returns
    -------
    Figure
        The Matplotlib figure object.
    """
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

    if not offscreen:
        fig.show()

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
    data: npt.NDArray[np.floating], clim: tuple[float, float] | None, cmap: str | None
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
) -> tuple[mtri.Triangulation, npt.NDArray[np.intp]]:
    """Build a triangulation for a specific region of the cerebellum flatmap.

    Parameters
    ----------
    cerebellum_geo : dict
        The cerebellum geometry object.
    region_key : str
        The key for the region of interest.
    subsampling : Literal["dense", "sparse"]
        The chosen subsampling of the cerebellar surface mesh.

    Returns
    -------
    tuple[mtri.Triangulation, npt.NDArray[np.intp]]
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
    data: npt.NDArray[np.floating],
    fwd_cerebellum_src: dict,
    cerebellum_geo: dict,
    subsampling: Literal["dense", "sparse"],
    smoothing_steps: int = 0,
) -> npt.NDArray[np.floating]:
    """Interpolate cerebellar data to specified subsampling and optionally smooth it.

    Calls `interpolate_cerebellum_data` to handle the interpolation via iterative
    nearest-neighbor averaging.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Data to be plotted on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices that are used in the
        forward solution for the cerebellar source space,
        i.e. `len(fwd_cerebellum_src['vertno'])`.
    fwd_cerebellum_src : dict
        The cerebellar source space used in the computation of the forward solution,
        typically `fwd['src'][1]`.
    cerebellum_geo : dict
        Cerebellum geometry object.
    subsampling : Literal["dense", "sparse"]
        Subsampling of the cerebellar surface mesh to which the data should be
        interpolated. Must match the subsampling used for calculating
        the forward solution.
    smoothing_steps : int, optional
        Number of smoothing iterations to apply to the interpolated data. Each
        iteration averages the value at each vertex with its neighbors. By default 0,
        which means no smoothing is applied.
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
    data_interpolated = interpolate_cerebellum_data(
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
        for vert in range(data_interpolated.shape[0]):
            data_interpolated[vert] = np.nanmean(
                data_interpolated[vert_to_neighbors[vert]]
            )

    return data_interpolated


def plot_sagittal(vol, only_show_midline=False, **kwargs):
    sag_ind = kwargs.get("sag_ind")
    title = kwargs.get("title")
    rr = kwargs.get("rr")
    nn = kwargs.get("nn")
    tris = kwargs.get("tris")
    cmap = kwargs.get("cmap")
    linewidth = kwargs.get("linewidth")
    if cmap is None:
        cmap = "gray_r"
    if linewidth is None:
        linewidth = 1.0
    fig, ax = plt.subplots(3, 2)
    fig.suptitle(title)

    if sag_ind is None:
        x_width = vol.shape[0]
        sag_ind = np.linspace(int(x_width * 0.1), int(x_width * 0.9), 6).astype(int)

    if only_show_midline:
        sag_ind = [sag_ind[3]]

    for c, slice_ind in enumerate(sag_ind):
        image = vol[slice_ind, :, :]
        plt.subplot(3, 2, c + 1)
        plt.imshow(image, cmap=cmap)

        if tris is not None:
            z_0 = slice_ind
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

        if nn is not None:
            ptsp = np.where(np.abs(rr[:, 0] - (slice_ind - 0.5)) < 1.0)[0]
            x_tp = rr[ptsp, 2]
            y_tp = rr[ptsp, 1]
            plt.quiver(
                x_tp, y_tp, nn[ptsp, 2], -nn[ptsp, 1], scale=1, scale_units="inches"
            )

    return fig, ax
