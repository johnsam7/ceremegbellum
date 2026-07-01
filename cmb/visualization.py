"""Visualization functions for cerebellar cortical data.

Provides plotting in normal 3D, inflated, and flatmap views using PyVista
and Matplotlib.
"""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: November, 2021
# License: MIT
# ---------------------------------------------------------------------------

import logging
from typing import Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
import numpy.typing as npt
import pyvista as pv

logger = logging.getLogger(__name__)


class MLabEmulator:
    def __init__(self):
        import pyvista as pv
        import pyvistaqt as pvqt

        self.pv = pv
        self.pvqt = pvqt

    def figure(self, bgcolor, fgcolor, size):
        self.plotter = self.pvqt.BackgroundPlotter()

    def triangular_mesh(self, x, y, z, triangles, scalars, colormap, clim):
        vertices = np.c_[x, y, z]

        faces = np.c_[np.full(len(triangles), 3), triangles]
        surf = self.pv.PolyData(vertices, faces)

        self.plotter.add_mesh(
            surf, opacity=1.0, scalars=scalars, cmap=colormap, clim=clim
        )

        return self.plotter

    def colorbar(self):
        pass

    def show(self):
        self.plotter.show()


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
        data_interpolated[nan_verts] = [
            np.nanmean(data_interpolated[vert_neighbor_group])
            for vert_neighbor_group in vert_neighbors
        ]
        nan_verts = np.where(np.isnan(data_interpolated))[0]

    logger.info("Cerebellum interpolation complete")

    return data_interpolated


def morph_cortex_data(
    cort_data: npt.NDArray[np.floating], cortex_src: dict, smoothing_steps: int
) -> npt.NDArray[np.floating]:
    """Interpolate and optionally smooth cortex data to dense surface vertices.

    Morphs from the vertices used (in forward solution) to the full cortical mesh.

    Parameters
    ----------
    cort_data : npt.NDArray[np.floating]
        Data to be morphed to the full cortical mesh. Should be an 1D array with length
        equal to the number of *used* vertices in the cortical source space
        (i.e., `cortex_src['nuse']`).
    cortex_src : dict
        Cortical source space dictionary, typically `fwd['src'][0]`.
    smoothing_steps : int
        Number of smoothing iterations to apply when morphing the data using
        `mne.morph._hemi_morph`.

    Returns
    -------
    npt.NDArray[np.floating]
        1D array of morphed cortical data on the full cortical mesh.
    """
    if cort_data.ndim != 1 or cort_data.shape[0] != len(cortex_src["vertno"]):
        raise ValueError(
            "cort_data must be a 1D array with length equal to the number of used "
            "vertices in the cortical source space (i.e., "
            "len(cortex_src['vertno']))."
        )
    logger.info(
        f"Morphing cortical data from {cort_data.shape[0]} vertices to full "
        f"cortical mesh with {cortex_src['np']} vertices using {smoothing_steps} "
        "smoothing steps."
    )
    morph = mne.morph._hemi_morph(
        cortex_src["tris"],
        np.arange(cortex_src["np"]),
        cortex_src["vertno"],
        smoothing_steps,
        maps=None,
        warn=True,
    )
    result = morph @ cort_data[:, None]

    return result


def combine_meshes(cerb_tris, cerb_rr, cerb_data, cort_tris, cort_rr, cort_data):
    """Combines cerebellum and cortex for plotting."""
    tris1 = cerb_tris
    tris2 = cort_tris + cerb_rr[:, 0].shape[0]
    new_rr = np.concatenate([cerb_rr, cort_rr])
    new_tris = np.concatenate([tris1, tris2])
    new_data = np.concatenate([cerb_data, cort_data])

    return new_rr, new_tris, new_data


def plot_normal(
    src_cerebellum: dict,
    cerebellum_data: npt.NDArray[np.floating],
    src_cortex: dict,
    cortex_data: npt.NDArray[np.floating],
    colormap: str,
    clim: tuple[float, float] | None,
    offscreen: bool = False,
    screenshot_fname: str | None = None,
) -> pv.Plotter:
    """Plot cerebellum and cortex in normal 3D view using PyVista.

    Parameters
    ----------
    src_cerebellum : dict
        Cerebellar source space dictionary, typically `fwd['src'][1]`.
    cerebellum_data : npt.NDArray[np.floating]
        Data to be visualized on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the dense triangulation of
        the cerebellar source space, i.e. `len(src_cerebellum['rr'])`.
    src_cortex : dict
        Cortical source space dictionary, typically `fwd['src'][0]`.
    cortex_data : npt.NDArray[np.floating]
        Data to be visualized on the cortex. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the dense triangulation of the
        cortical source space, i.e. `len(src_cortex['rr'])`.
    colormap : str
        Color map for 3D plots.
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
    if clim is None:
        clim = _determine_clim(cerebellum_data, cortex_data)

    plotter = pv.Plotter(window_size=[1200, 1200], off_screen=offscreen)
    # Ignoring warning because my pyright is confused.
    plotter.set_background(color="white")  # pyright: ignore[reportCallIssue]

    cerebellum_mesh = _make_cerebellum_visualization(src_cerebellum, cerebellum_data)
    plotter.add_mesh(
        cerebellum_mesh,
        scalars="scalars",
        cmap=colormap,
        scalar_bar_args={"color": "black"},
    )

    if cortex_data is not None:
        cortex_mesh = _make_cortex_visualization(src_cortex, cortex_data)
        plotter.add_mesh(cortex_mesh, scalars="scalars", cmap=colormap)

    plotter.camera.position = (0, -1, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.focal_point = cerebellum_mesh.center
    plotter.reset_camera()  # pyright: ignore[reportCallIssue]

    if screenshot_fname is not None:
        plotter.screenshot(screenshot_fname)
        logger.info(f"Saved normal view to {screenshot_fname}")

    plotter.show()

    return plotter


def _determine_clim(
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


def _make_cortex_visualization(
    src_cortex: dict, data: npt.NDArray[np.floating]
) -> pv.PolyData:
    """Create a PyVista PolyData object for the cortex visualization.

    Makes a triangular mesh with a scalar value in each vertex using all vertices in
    the cortical source space.

    Parameters
    ----------
    src_cortex : dict
        The cortical source space dictionary, typically `fwd['src'][0]`.
    data : npt.NDArray[np.floating]
        Data to be visualized on the cortex. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the cortical source space, i.e.
        `len(src_cortex['rr'])`.
    """
    verts = src_cortex["rr"]
    faces = src_cortex["tris"]

    pv_cortex_faces = np.column_stack([np.full(len(faces), 3), faces])
    mesh = pv.PolyData(verts, pv_cortex_faces)
    mesh.point_data["scalars"] = data

    return mesh


def _make_cerebellum_visualization(
    src_cerebellum: dict, data: npt.NDArray[np.floating]
) -> pv.PolyData:
    """Create a PyVista PolyData object for the cerebellum visualization.

    Makes a triangular mesh with a scalar value in each vertex using all vertices in
    the cerebellar source space.

    Parameters
    ----------
    src_cerebellum : dict
        The cerebellar source space dictionary, typically `fwd['src'][1]`.
    data : npt.NDArray[np.floating]
        Data to be visualized on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the dense triangulation of
        the cerebellar source space, i.e. `len(src_cerebellum['rr'])`.
    """
    verts = src_cerebellum["rr"]
    faces = src_cerebellum["tris"]

    # Add a column of 3s to tell PyVista that these are triangles (3 vertices per face).
    pv_faces = np.column_stack([np.full(len(faces), 3), faces])

    mesh = pv.PolyData(verts, pv_faces)
    mesh.point_data["scalars"] = data

    return mesh


def plot_inflated(
    cerebellum_geo: dict,
    cerebellum_data: npt.NDArray[np.floating],
    subsampling: Literal["dense", "sparse"],
    colormap: str,
    clim: tuple[float, float] | None,
    offscreen: bool = False,
    screenshot_fname: str | None = None,
) -> pv.Plotter:
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
    colormap : str
        Color map for 3D plots.
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
        cmap=colormap,
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

    plotter.show()

    return plotter


def plot_flatmap(cerebellum_geo, estimate_smoothed, colormap, cmap_lims, sub_sampling):
    import matplotlib.colors as colors
    import matplotlib.tri as mtri

    figures = list()

    def truncate_colormap(colormap, minval=0.0, maxval=1.0, n=500):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            f"trunc({colormap.name},{minval:.2f},{maxval:.2f})",
            colormap(np.linspace(minval, maxval, n)),
        )
        return new_cmap

    if np.min(estimate_smoothed) >= 0:
        red_cmap = truncate_colormap(plt.get_cmap(colormap), 0.5, 1.0)
        color_levels = np.ones((cmap_lims[0] + 1, 4))
        color_levels = np.vstack(
            (color_levels, red_cmap(np.linspace(0, 1, cmap_lims[1] - cmap_lims[0])))
        )
        color_levels = np.vstack(
            (
                color_levels,
                np.repeat(
                    red_cmap([1.0]).reshape(1, 4), repeats=100 - cmap_lims[1], axis=0
                ),
            )
        )
        cmap_real = red_cmap
    else:
        blue_cmap = truncate_colormap(plt.get_cmap(colormap), 0.0, 0.5)
        red_cmap = truncate_colormap(plt.get_cmap(colormap), 0.5, 1.0)
        color_levels = np.repeat(
            blue_cmap([0.0]).reshape(1, 4), repeats=100 - cmap_lims[1], axis=0
        )
        color_levels = np.vstack(
            (color_levels, blue_cmap(np.linspace(0, 1, cmap_lims[1] - cmap_lims[0])))
        )
        color_levels = np.vstack((color_levels, np.ones((cmap_lims[0] + 1, 4))))
        color_levels = np.vstack((color_levels, np.ones((cmap_lims[0], 4))))
        color_levels = np.vstack(
            (color_levels, red_cmap(np.linspace(0, 1, cmap_lims[1] - cmap_lims[0])))
        )
        color_levels = np.vstack(
            (
                color_levels,
                np.repeat(
                    red_cmap([1.0]).reshape(1, 4), repeats=100 - cmap_lims[1], axis=0
                ),
            )
        )

    max_abs = np.max(np.abs(estimate_smoothed))
    if np.min(estimate_smoothed) >= 0:
        levels = np.linspace(0, max_abs, 101)
    else:
        levels = np.linspace(-max_abs, max_abs, 201)

    font = {"weight": "normal", "size": 8}
    plt.rc("font", **font)
    flat_fig = plt.figure(dpi=300, figsize=(7, 5.5))

    for flatmap in cerebellum_geo["flatmap_outlines"]:
        lin = plt.plot(
            -flatmap[:, 0],
            flatmap[:, 1],
            linestyle="--",
            linewidth=0.4,
            c="k",
            alpha=1.0,
        )[0]  # minus x-coord for keeping in neurological coordinates

    for key in list(cerebellum_geo["flatmap_inds"].keys()):
        dw_inds = np.where(
            np.isin(
                cerebellum_geo["dw_data"][sub_sampling],
                cerebellum_geo["flatmap_inds"][key],
            )
        )[0]
        dw_flatinds = cerebellum_geo["dw_data"][sub_sampling][dw_inds]
        flat_verts = cerebellum_geo["verts_flatmap"][dw_flatinds, :]

        ind_map = np.zeros(cerebellum_geo["dw_data"][sub_sampling].shape[0])
        ind_map[:] = np.nan
        ind_map[dw_inds] = np.linspace(0, len(dw_inds) - 1, len(dw_inds)).astype(int)
        tris_flat = cerebellum_geo["dw_data"][sub_sampling + "_tris"][
            np.where(
                np.isin(cerebellum_geo["dw_data"][sub_sampling + "_tris"], dw_inds).all(
                    axis=1
                )
            )[0],
            :,
        ]
        tris_flat = ind_map[tris_flat].astype(int)
        estimate_flat_all = estimate_smoothed[dw_inds]
        triang = mtri.Triangulation(
            -flat_verts[:, 0], flat_verts[:, 1], tris_flat
        )  # minus x-coord for keeping in neurological coordinates
        triconf = lin.axes.tricontourf(
            triang, estimate_flat_all, colors=color_levels, levels=levels
        )  # flatmap_cmap=hot_truncated_cmap)

    if np.min(levels) < 0:
        cbar = flat_fig.colorbar(
            triconf,
            ticks=[
                levels[0],
                levels[100 - cmap_lims[1]],
                levels[100 - cmap_lims[0]],
                0,
                levels[100 + cmap_lims[0]],
                levels[100 + cmap_lims[1]],
                levels[len(levels) - 1],
            ],
        )
        min_lev = str(levels[0])[0:4] + str(levels[0])[str(levels[0]).find("e") :]
        min_sat = (
            str(levels[100 - cmap_lims[1]])[0:4]
            + str(levels[100 - cmap_lims[1]])[
                str(levels[100 - cmap_lims[1]]).find("e") :
            ]
        )
        min_thresh = (
            str(levels[100 - cmap_lims[0]])[0:4]
            + str(levels[100 - cmap_lims[0]])[
                str(levels[100 - cmap_lims[0]]).find("e") :
            ]
        )
        max_lev = (
            str(levels[len(levels) - 1])[0:4]
            + str(levels[len(levels) - 1])[str(levels[len(levels) - 1]).find("e") :]
        )
        max_sat = (
            str(levels[100 + cmap_lims[1]])[0:4]
            + str(levels[100 + cmap_lims[1]])[
                str(levels[100 + cmap_lims[1]]).find("e") :
            ]
        )
        max_thresh = (
            str(levels[100 + cmap_lims[0]])[0:4]
            + str(levels[100 + cmap_lims[0]])[
                str(levels[100 + cmap_lims[0]]).find("e") :
            ]
        )
        cbar.ax.set_yticklabels(
            [min_lev, min_sat, min_thresh, "0", max_thresh, max_sat, max_lev]
        )
    else:
        cbar = flat_fig.colorbar(
            triconf,
            ticks=[
                0,
                levels[cmap_lims[0]],
                levels[cmap_lims[1]],
                levels[len(levels) - 1],
            ],
        )
        max_lev = (
            str(levels[len(levels) - 1])[0:4]
            + str(levels[len(levels) - 1])[str(levels[len(levels) - 1]).find("e") :]
        )
        max_sat = (
            str(levels[cmap_lims[1]])[0:4]
            + str(levels[cmap_lims[1]])[str(levels[cmap_lims[1]]).find("e") :]
        )
        max_thresh = (
            str(levels[cmap_lims[0]])[0:4]
            + str(levels[cmap_lims[0]])[str(levels[cmap_lims[0]]).find("e") :]
        )
        cbar.ax.set_yticklabels(["0", max_thresh, max_sat, max_lev])

    ant_lob = np.array(
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
    )
    crusII_left = np.array([[-165, 257], [-126, 260], [-80, 300]])
    crusII_right = np.array([[96, 313], [230, 148]])
    lobVIIb_left = np.array([[-239, -49], [-178, -117]])
    lobVIIb_right = np.array([[255, -211], [244, -119], [265, -75], [293, -71]])

    for border_line in [
        ant_lob,
        crusII_left,
        crusII_right,
        lobVIIb_left,
        lobVIIb_right,
    ]:
        plt.plot(
            -border_line[:, 0],
            border_line[:, 1],
            linestyle="--",
            linewidth=0.4,
            c="k",
            alpha=1.0,
        )  # minus x-coord for keeping in neurological coordinates
    plt.gca().set_aspect("equal")

    # place text boxes outlining anatomical landmarks
    axis = flat_fig.axes[0]
    text_params = {"fontsize": 8, "verticalalignment": "top"}

    axis.text(-610, 1175, " Lobules I-V \n (anterior lobe)", **text_params)
    axis.text(-490, 840, "Lobule VI", **text_params)
    axis.text(-450, 441, "Crus I", **text_params)
    axis.text(-700, 100, " Crus II/\n Lobule VIIb", **text_params)
    axis.text(-740, -200, "Lobule VIII", **text_params)
    axis.text(-670, -590, " Lobule IX \n (tonsil)", **text_params)
    axis.text(-370, -670, " Lobule X \n (flocculus)", **text_params)
    axis.text(40, -710, "Inferior vermis", **text_params)
    axis.text(-380, 1480, "Left", fontweight="bold", **text_params)
    axis.text(200, 1480, "Right", fontweight="bold", **text_params)

    arrow_params = {
        "head_width": 20,
        "head_length": 20,
        "linewidth": 0.5,
        "fc": "k",
        "ec": "k",
    }

    axis.arrow(-350, -580, 82, 64, **arrow_params)
    axis.arrow(-230, -660, 80, 45, **arrow_params)
    axis.arrow(20, -750, 0, 130, **arrow_params)

    flat_fig.patch.set_visible(False)
    axis.axis("off")
    plt.show()
    figures.append(flat_fig)

    return figures


def plot_cerebellum_data(
    data: npt.NDArray[np.floating],
    fwd_src: mne.SourceSpaces,
    org_src: mne.SourceSpaces,
    cerebellum_geo: dict,
    cort_data: npt.NDArray[np.floating] | None = None,
    flatmap_cmap: str = "bwr",
    mayavi_cmap: str | None = None,
    n_smoothing_steps: int = 0,
    view: Literal["all", "normal", "inflated", "flatmap"] | None = None,
    sub_sampling: Literal["dense", "sparse"] = "sparse",
    cmap_lims: tuple = (1, 98),
    clim=None,
) -> tuple:
    """Plot data on the cerebellar cortical surface.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Data to be plotted on the cerebellum. Should have shape (n_vertices,),
        where n_vertices is the number of vertices in the cerebellar source space.
    fwd_src : mne.SourceSpaces
        The source space used in the computation of the forward solution, fwd['src'].
    org_src : mne.SourceSpaces
        Full surface source space for both cortex and cerebellum.
    cerebellum_geo : dict
        Cerebellum 3D geometry object.
    cort_data : npt.NDArray[np.floating] | None, optional
        Data to be plotted for each vertex in the cortical source space.
        Should have shape (n_vertices,). By default None.
    flatmap_cmap : str, optional
        Color map for 2D plots, by default "bwr"
    mayavi_cmap : str | None, optional
        Color map for 3D plots, by default None
    n_smoothing_steps : int, optional
        Number of smoothing iterations, by default 0
    view : Literal["all", "normal", "inflated", "flatmap"] | None, optional
        Which views to show.
    sub_sampling : Literal["dense", "sparse"], optional
        Sub-sampling of the data provided, by default "sparse"
    cmap_lims : tuple, optional
        Colormap limits, where first element is the lower bound and the
        second element is the upper bound, by default (1, 98)
    clim : _type_, optional
        _description_, by default None

    Returns
    -------
    tuple
        Tuple of figures, where each figure is either a PyVista plotter or a
        Matplotlib figure.
    """
    mlab = MLabEmulator()

    if cort_data is not None and cort_data.shape[0] != fwd_src[0]["nuse"]:
        raise ValueError(
            "cort_data and src[0]['nuse'] must have the same number of elements."
        )

    # Indices of the vertices in the cerebellar source space that are used in the
    # forward solution. Data is provided for these vertices.
    data_indices = fwd_src[1]["vertno"]
    # Help IDE type checkers with assertion.
    assert isinstance(data_indices, np.ndarray), (
        "fwd_src[1]['vertno'] must be a numpy array."
    )

    estimate_interpolated = interpolate_cerebellum_data(
        data,
        data_indices=data_indices,
        subsampling=sub_sampling,
        cerebellum_geo=cerebellum_geo,
    )

    src_cortex = fwd_src[0]
    src_cerebellum = fwd_src[1]

    cort_data_full_mesh = None
    if cort_data is not None and view in ["all", "normal"]:
        assert isinstance(src_cortex, dict), "fwd_src[0] must be a dictionary."
        cort_data_full_mesh = morph_cortex_data(
            cort_data, src_cortex, n_smoothing_steps
        )

    if mayavi_cmap is None:
        if cort_data is None:
            if np.min(estimate_interpolated) < 0:
                mayavi_cmap = "bwr"
            else:
                mayavi_cmap = "OrRd"
        else:
            if np.min(np.concatenate((estimate_interpolated, cort_data))) < 0:
                mayavi_cmap = "bwr"
            else:
                mayavi_cmap = "OrRd"

    for step in range(n_smoothing_steps):
        print("Step " + str(step))
        for vert in range(estimate_interpolated.shape[0]):
            estimate_interpolated[vert] = np.nanmean(
                estimate_interpolated[
                    cerebellum_geo["dw_data"][sub_sampling + "_vert_to_neighbor"][vert]
                ]
            )

    figures = []

    assert isinstance(src_cerebellum, dict), "fwd_src[1] must be a dictionary."
    assert isinstance(src_cortex, dict), "fwd_src[0] must be a dictionary."

    if view in ["all", "normal"]:
        assert cort_data is not None, "cort_data must be provided for normal view."
        assert cort_data_full_mesh is not None, (
            "cort_data_full_mesh must be computed for normal view."
        )
        plotter = plot_normal(
            src_cerebellum=src_cerebellum,
            cerebellum_data=estimate_interpolated,
            src_cortex=src_cortex,
            cortex_data=cort_data_full_mesh,
            colormap=mayavi_cmap,
            clim=None,
            offscreen=False,
            screenshot_fname=None,
        )
        figures.append(plotter)

    if view in ["all", "inflated"]:
        plotter = plot_inflated(
            cerebellum_geo,
            cerebellum_data=estimate_interpolated,
            subsampling=sub_sampling,
            colormap=mayavi_cmap,
            clim=None,
            offscreen=False,
            screenshot_fname=None,
        )
        figures.append(plotter)

    if view in ["all", "flatmap"]:
        figures += plot_flatmap(
            cerebellum_geo, estimate_interpolated, flatmap_cmap, cmap_lims, sub_sampling
        )

    return figures


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
