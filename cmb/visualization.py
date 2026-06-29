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
from mne.morph import _hemi_morph

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

    return data_interpolated


def one_pass_cortex_smoothing(cort_data, org_src, src_cort, smoothing_steps):
    morph = _hemi_morph(
        org_src[0]["tris"],
        np.arange(org_src[0]["np"]),
        org_src[0]["vertno"],
        smoothing_steps,
        maps=None,
        warn=True,
    )
    return morph @ cort_data[:, None], org_src[0]["tris"]


def combine_meshes(cerb_tris, cerb_rr, cerb_data, cort_tris, cort_rr, cort_data):
    """Combines cerebellum and cortex for plotting."""
    tris1 = cerb_tris
    tris2 = cort_tris + cerb_rr[:, 0].shape[0]
    new_rr = np.concatenate([cerb_rr, cort_rr])
    new_tris = np.concatenate([tris1, tris2])
    new_data = np.concatenate([cerb_data, cort_data])

    return new_rr, new_tris, new_data


def plot_normal(
    mlab,
    src_cerb,
    cort_data,
    org_src,
    src_cort,
    estimate_smoothed,
    cerebellum_geo,
    sub_sampling,
    colormap,
    tris_frame,
    cort_full_mantle,
    clim=None,
):

    mlab.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), size=(1200, 1200))
    if cort_data is None:
        return [
            mlab.triangular_mesh(
                src_cerb["rr"][:, 0],
                src_cerb["rr"][:, 1],
                src_cerb["rr"][:, 2],
                cerebellum_geo["dw_data"][sub_sampling + "_tris"],
                scalars=estimate_smoothed,
                colormap=colormap,
                clim=clim,
            )
        ]
    else:
        if org_src[0]["use_tris"] is not None:
            rr_cx = src_cort["rr"][org_src[0]["vertno"], :]
        else:
            rr_cx = src_cort["rr"]

        rr, tris, data = combine_meshes(
            cerebellum_geo["dw_data"][sub_sampling + "_tris"],
            src_cerb["rr"],
            estimate_smoothed,
            tris_frame,
            rr_cx,
            np.concatenate(cort_full_mantle),
        )
        return [
            mlab.triangular_mesh(
                rr[:, 0],
                rr[:, 1],
                rr[:, 2],
                tris,
                scalars=data,
                colormap=colormap,
                clim=clim,
            )
        ]


def plot_inflated(
    mlab, estimate_smoothed, cerebellum_geo, sub_sampling, colormap, clim=None
):
    figures = list()
    mlab.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), size=(1200, 1200))
    verts = cerebellum_geo["verts_inflated_fs"]
    dw_data = cerebellum_geo["dw_data"][sub_sampling]
    inflated_fig = mlab.triangular_mesh(
        verts[dw_data, 0],
        verts[dw_data, 1],
        verts[dw_data, 2],
        cerebellum_geo["dw_data"][sub_sampling + "_tris"],
        scalars=estimate_smoothed,
        colormap=colormap,
        clim=clim,
    )

    mlab.colorbar()
    figures.append(inflated_fig)
    return figures


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
    sub_sampling: Literal["dense", "sparse", "full"] = "sparse",
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
    sub_sampling : Literal["dense", "sparse", "full"], optional
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

    src_cerb = fwd_src[1]
    estimate_smoothed = one_pass_cerebellum_smoothing(
        data, src_cerb, cerebellum_geo, sub_sampling
    )

    src_cort = fwd_src[0]
    cort_full_mantle = None
    tris_frame = None
    if cort_data is not None and view in ["all", "normal"]:
        cort_full_mantle, tris_frame = one_pass_cortex_smoothing(
            cort_data, fwd_src, src_cort, n_smoothing_steps
        )

    if mayavi_cmap is None:
        if cort_data is None:
            if np.min(estimate_smoothed) < 0:
                mayavi_cmap = "bwr"
            else:
                mayavi_cmap = "OrRd"
        else:
            if np.min(np.concatenate((estimate_smoothed, cort_data))) < 0:
                mayavi_cmap = "bwr"
            else:
                mayavi_cmap = "OrRd"

    for step in range(n_smoothing_steps):
        print("Step " + str(step))
        for vert in range(estimate_smoothed.shape[0]):
            estimate_smoothed[vert] = np.nanmean(
                estimate_smoothed[
                    cerebellum_geo["dw_data"][sub_sampling + "_vert_to_neighbor"][vert]
                ]
            )

    figures = []

    if view in ["all", "normal"]:
        figures += plot_normal(
            mlab,
            src_cerb,
            cort_data,
            org_src,
            src_cort,
            estimate_smoothed,
            cerebellum_geo,
            sub_sampling,
            mayavi_cmap,
            tris_frame,
            cort_full_mantle,
            clim=clim,
        )

    if view in ["all", "inflated"]:
        figures += plot_inflated(
            mlab,
            estimate_smoothed,
            cerebellum_geo,
            sub_sampling,
            mayavi_cmap,
            clim=clim,
        )

    if view in ["all", "flatmap"]:
        figures += plot_flatmap(
            cerebellum_geo, estimate_smoothed, flatmap_cmap, cmap_lims, sub_sampling
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
