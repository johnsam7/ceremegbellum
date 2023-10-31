#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: November, 2021
# License: MIT
# ---------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

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

        self.plotter.add_mesh(surf, opacity=1.0,
                              scalars=scalars, cmap=colormap, clim=clim)

        return self.plotter

    def colorbar(self):
        pass

    def show(self):
        self.plotter.show()


def one_pass_cerebellum_smoothing(data, src_cerb, cerebellum_geo, sub_sampling):
    print('Smoothing...')
    estimate_smoothed = np.zeros(cerebellum_geo['dw_data'][sub_sampling+'_verts'].shape[0])
    estimate_smoothed[:] = np.nan
    estimate_smoothed[src_cerb['vertno']] = data
    nan_verts = np.where(np.isnan(estimate_smoothed))[0]

    while len(nan_verts) > 0:
        vert_neighbors = [cerebellum_geo['dw_data'][sub_sampling+'_vert_to_neighbor'][ind] for ind in nan_verts]
        estimate_smoothed[nan_verts] = [np.nanmean(estimate_smoothed[vert_neighbor_group]) for vert_neighbor_group in vert_neighbors]
        nan_verts = np.where(np.isnan(estimate_smoothed))[0]
    return estimate_smoothed


def one_pass_cortex_smoothing(cort_data, org_src, src_cort, smoothing_steps):
    morph = mne.morph._hemi_morph(
        org_src[0]['tris'],
        np.arange(org_src[0]["np"]),
        org_src[0]['vertno'],
        smoothing_steps,
        maps=None,
        warn=True,
    )
    return morph @ cort_data[:, None], org_src[0]['tris']


def plot_normal(mlab, src_cerb, cort_data, org_src, src_cort, estimate_smoothed,
                cerebellum_geo, sub_sampling, colormap, tris_frame, cort_full_mantle, clim=None):

    figures = list()
    if cort_data is None:
        mlab.figure(bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.), size=(1200, 1200))
        cereb_fig = mlab.triangular_mesh(
                src_cerb['rr'][:, 0], src_cerb['rr'][:, 1], src_cerb['rr'][:, 2],
                cerebellum_geo['dw_data'][sub_sampling+'_tris'],
                scalars=estimate_smoothed, colormap=colormap, clim=clim)
        mlab.colorbar()
        figures.append(cereb_fig)
    else:
        if org_src[0]['use_tris'] is not None:
            rr_cx = src_cort['rr'][org_src[0]['vertno'], :]
        else:
            rr_cx = src_cort['rr']

        x = src_cerb['rr'][:, 0]
        y = src_cerb['rr'][:, 1]
        z = src_cerb['rr'][:, 2]
        tris = cerebellum_geo['dw_data'][sub_sampling+'_tris']
        plot_data = estimate_smoothed

        x2 = rr_cx[:, 0]
        y2 = rr_cx[:, 1]
        z2 = rr_cx[:, 2]
        tris2 = tris_frame+x.shape[0]
        plot_data2 = np.concatenate(cort_full_mantle)

        new_x = np.concatenate([x, x2])
        new_y = np.concatenate([y, y2])
        new_z = np.concatenate([z, z2])

        new_tris = np.concatenate([tris, tris2])
        new_plot_data = np.concatenate([plot_data, plot_data2])

        mlab.figure(bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.), size=(1200, 1200))
        full_fig = mlab.triangular_mesh(new_x,new_y,new_z,new_tris,scalars=new_plot_data, colormap=colormap, clim=clim)
        figures.append(full_fig)
    return figures


def plot_inflated(mlab, estimate_smoothed, cerebellum_geo,
                  sub_sampling, colormap, clim=None):
    figures = list()
    mlab.figure(bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.), size=(1200, 1200))
    verts = cerebellum_geo['verts_inflated_fs']
    dw_data = cerebellum_geo['dw_data'][sub_sampling]
    inflated_fig = mlab.triangular_mesh(
            verts[dw_data, 0], verts[dw_data, 1], verts[dw_data, 2],
            cerebellum_geo['dw_data'][sub_sampling+'_tris'],
            scalars=estimate_smoothed, colormap=colormap, clim=clim)

    mlab.colorbar()
    figures.append(inflated_fig)
    return figures


def plot_flatmap(cerebellum_geo, estimate_smoothed, colormap,
                 cmap_lims, sub_sampling):
    import matplotlib.colors as colors
    import matplotlib.tri as mtri

    figures = list()

    def truncate_colormap(colormap, minval=0.0, maxval=1.0, n=500):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            f'trunc({colormap.name},{minval:.2f},{maxval:.2f})',
            colormap(np.linspace(minval, maxval, n)))
        return new_cmap

    if np.min(estimate_smoothed) >= 0:
        red_cmap = truncate_colormap(plt.get_cmap(colormap), 0.5, 1.)
        color_levels = np.ones((cmap_lims[0]+1, 4))
        color_levels = np.vstack((color_levels, red_cmap(np.linspace(0, 1, cmap_lims[1]-cmap_lims[0]))))
        color_levels = np.vstack((color_levels, np.repeat(red_cmap([1.]).reshape(1,4), repeats=100-cmap_lims[1], axis=0)))
        cmap_real=red_cmap
    else:
        blue_cmap = truncate_colormap(plt.get_cmap(colormap), 0.0, 0.5)
        red_cmap = truncate_colormap(plt.get_cmap(colormap), 0.5, 1.)
        color_levels = np.repeat(blue_cmap([0.]).reshape(1,4), repeats=100-cmap_lims[1], axis=0)
        color_levels = np.vstack((color_levels, blue_cmap(np.linspace(0, 1, cmap_lims[1]-cmap_lims[0]))))
        color_levels = np.vstack((color_levels, np.ones((cmap_lims[0]+1, 4))))
        color_levels = np.vstack((color_levels, np.ones((cmap_lims[0], 4))))
        color_levels = np.vstack((color_levels, red_cmap(np.linspace(0, 1, cmap_lims[1]-cmap_lims[0]))))
        color_levels = np.vstack((color_levels, np.repeat(red_cmap([1.]).reshape(1,4), repeats=100-cmap_lims[1], axis=0)))

    max_abs = np.max(np.abs(estimate_smoothed))
    if np.min(estimate_smoothed) >= 0:
        levels = np.linspace(0, max_abs, 101)
    else:
        levels = np.linspace(-max_abs, max_abs, 201)

    font = {'weight' : 'normal', 'size'   : 8}
    plt.rc('font', **font)
    flat_fig = plt.figure(dpi = 300, figsize = (7, 5.5))

    for flatmap in cerebellum_geo['flatmap_outlines']:
        lin = plt.plot(-flatmap[:, 0], flatmap[:, 1], linestyle='--', linewidth=0.4, c='k', alpha=1.0)[0] # minus x-coord for keeping in neurological coordinates

    for key in list(cerebellum_geo['flatmap_inds'].keys()):
        dw_inds = np.where(np.isin(cerebellum_geo['dw_data'][sub_sampling], cerebellum_geo['flatmap_inds'][key]))[0]
        dw_flatinds = cerebellum_geo['dw_data'][sub_sampling][dw_inds]
        flat_verts = cerebellum_geo['verts_flatmap'][dw_flatinds, :]

        ind_map = np.zeros(cerebellum_geo['dw_data'][sub_sampling].shape[0])
        ind_map[:] = np.nan
        ind_map[dw_inds] = np.linspace(0, len(dw_inds)-1, len(dw_inds)).astype(int)
        tris_flat = cerebellum_geo['dw_data'][sub_sampling+'_tris'][np.where(np.isin(cerebellum_geo['dw_data'][sub_sampling+'_tris'],
                                                                              dw_inds).all(axis=1))[0], :]
        tris_flat = ind_map[tris_flat].astype(int)
        estimate_flat_all = estimate_smoothed[dw_inds]
        triang = mtri.Triangulation(-flat_verts[:,0], flat_verts[:,1], tris_flat) # minus x-coord for keeping in neurological coordinates
        triconf = lin.axes.tricontourf(triang, estimate_flat_all, colors=color_levels, levels=levels)# flatmap_cmap=hot_truncated_cmap) 


    if np.min(levels) < 0:
        cbar = flat_fig.colorbar(triconf, ticks=[levels[0], levels[100-cmap_lims[1]], levels[100-cmap_lims[0]],
                                                 0, levels[100+cmap_lims[0]], levels[100+cmap_lims[1]], levels[len(levels)-1]])
        min_lev = str(levels[0])[0:4]+str(levels[0])[str(levels[0]).find('e'):]
        min_sat = str(levels[100-cmap_lims[1]])[0:4]+str(levels[100-cmap_lims[1]])[str(levels[100-cmap_lims[1]]).find('e'):]
        min_thresh = str(levels[100-cmap_lims[0]])[0:4]+str(levels[100-cmap_lims[0]])[str(levels[100-cmap_lims[0]]).find('e'):]
        max_lev = str(levels[len(levels)-1])[0:4]+str(levels[len(levels)-1])[str(levels[len(levels)-1]).find('e'):]
        max_sat = str(levels[100+cmap_lims[1]])[0:4]+str(levels[100+cmap_lims[1]])[str(levels[100+cmap_lims[1]]).find('e'):]
        max_thresh = str(levels[100+cmap_lims[0]])[0:4]+str(levels[100+cmap_lims[0]])[str(levels[100+cmap_lims[0]]).find('e'):]
        cbar.ax.set_yticklabels([min_lev, min_sat, min_thresh, '0', max_thresh, max_sat, max_lev])  
    else:
        cbar = flat_fig.colorbar(triconf, ticks=[0, levels[cmap_lims[0]], 
                                                 levels[cmap_lims[1]], levels[len(levels)-1]])
        max_lev = str(levels[len(levels)-1])[0:4]+str(levels[len(levels)-1])[str(levels[len(levels)-1]).find('e'):]
        max_sat = str(levels[cmap_lims[1]])[0:4]+str(levels[cmap_lims[1]])[str(levels[cmap_lims[1]]).find('e'):]
        max_thresh = str(levels[cmap_lims[0]])[0:4]+str(levels[cmap_lims[0]])[str(levels[cmap_lims[0]]).find('e'):]
        cbar.ax.set_yticklabels(['0', max_thresh, max_sat, max_lev])  

    ant_lob = np.array([[-110, 918], [-86,925], [-52, 935], [-16, 942],
                        [23, 965], [57, 980], [78, 983], [123, 966]])
    crusII_left = np.array([[-165, 257], [-126, 260], [-80, 300]])
    crusII_right = np.array([[96, 313], [230, 148]])
    lobVIIb_left = np.array([[-239, -49], [-178, -117]])
    lobVIIb_right = np.array([[255, -211], [244, -119], [265, -75], [293, -71]])

    for border_line in [ant_lob, crusII_left, crusII_right, lobVIIb_left, lobVIIb_right]:
        plt.plot(-border_line[:,0], border_line[:,1], linestyle='--', linewidth=0.4, c='k', alpha=1.0) # minus x-coord for keeping in neurological coordinates
    plt.gca().set_aspect('equal')

    # place text boxes outlining anatomical landmarks
    axis = flat_fig.axes[0]
    text_params = {'fontsize': 8, 'verticalalignment': 'top'}

    axis.text(-610, 1175, ' Lobules I-V \n (anterior lobe)', **text_params)
    axis.text(-490, 840, 'Lobule VI', **text_params)
    axis.text(-450, 441, 'Crus I', **text_params)
    axis.text(-700, 100, ' Crus II/\n Lobule VIIb', **text_params)
    axis.text(-740, -200, 'Lobule VIII', **text_params)
    axis.text(-670, -590, ' Lobule IX \n (tonsil)', **text_params)
    axis.text(-370, -670, ' Lobule X \n (flocculus)', **text_params)
    axis.text(40, -710, 'Inferior vermis', **text_params)
    axis.text(-380, 1480, 'Left', fontweight='bold', **text_params)
    axis.text(200, 1480, 'Right', fontweight='bold', **text_params)

    arrow_params = {'head_width': 20, 'head_length': 20,
                    'linewidth': 0.5, 'fc': 'k', 'ec': 'k'}

    axis.arrow(-350, -580, 82, 64, **arrow_params)
    axis.arrow(-230, -660, 80, 45, **arrow_params)
    axis.arrow(20, -750, 0, 130, **arrow_params)

    flat_fig.patch.set_visible(False)
    axis.axis('off')
    plt.show()
    figures.append(flat_fig)

    return figures


def plot_cerebellum_data(data, fwd_src, org_src, cerebellum_geo,
                         cort_data=None, flatmap_cmap='bwr',
                         mayavi_cmap=None, smoothing_steps=0, view='all',
                         sub_sampling='sparse', cmap_lims=[1, 98], clim=None):
    """Plots data on the cerebellar cortical surface. Requires cerebellum
    geometry file to be downloaded.

    Parameters
    ----------
    data : array, shape (n_vertices)
        Cerebellar data.
    fwd_src : MNE SourceSpaces
        The source space used in the computation of the forward solution.
    org_src: MNE SourceSpaces
        Full surface fource space for both coreex and cerebellum.
    cerebellum_geo : dict
        Cerebellum 3D geometry object
    cort_data : array
        Cortex data
    flatmap_cmap : string
        Color map for 2D plots
    mayavi_cmap : string
        Color map for 3D plots
    smoothing_steps:
        Cerebellum smoothing iterations
    view: "all" | "normal" | "inflated" | "flatmap"
        Which views to show. If view='all', then all
        (normal, inflated and flamap) are shown.
    sub_sampling : string
        'dense', 'sparse', 'full'. Has to coorespond to sub sampling of
        other data provided.
    cmap_lims : list
        Colormap limits, where first element is the lower bound and the
        second element is the upper bound.

    Returns
    -------
    figures: list
        List containing Figure objects.

    """

    mlab = MLabEmulator()

    if cort_data is not None:
        assert cort_data.shape[0] == fwd_src[0]['nuse'], 'cort_data and src[0][\'nuse\'] must have the same number of elements.'

    src_cerb = fwd_src[1]
    estimate_smoothed = one_pass_cerebellum_smoothing(
                            data, src_cerb, cerebellum_geo, sub_sampling)

    src_cort = fwd_src[0]
    if cort_data is not None:
        cort_full_mantle, tris_frame = one_pass_cortex_smoothing(
                                            cort_data, org_src, src_cort,
                                            smoothing_steps)

    if mayavi_cmap is None:
        if cort_data is None:
            if np.min(estimate_smoothed) < 0:
                mayavi_cmap = 'bwr'
            else:
                mayavi_cmap = 'OrRd'
        else:
            if np.min(np.concatenate((estimate_smoothed, cort_data))) < 0:
                mayavi_cmap = 'bwr'
            else:
                mayavi_cmap = 'OrRd'

    for step in range(smoothing_steps):
        print('Step '+str(step))
        for vert in range(estimate_smoothed.shape[0]):
            estimate_smoothed[vert] = np.nanmean(estimate_smoothed[cerebellum_geo['dw_data'][sub_sampling+'_vert_to_neighbor'][vert]])

    figures = []

    if view in ['all', 'normal']:
        figures += plot_normal(mlab, src_cerb, cort_data, org_src,
                               src_cort, estimate_smoothed,
                               cerebellum_geo, sub_sampling, mayavi_cmap,
                               tris_frame, cort_full_mantle, clim=clim)

    if view in ['all', 'inflated']:
        figures += plot_inflated(mlab, estimate_smoothed, cerebellum_geo,
                                 sub_sampling, mayavi_cmap, clim=clim)

    if view in ['all', 'flatmap']:
        figures += plot_flatmap(cerebellum_geo, estimate_smoothed,
                                flatmap_cmap, cmap_lims, sub_sampling)

    return figures

# TBD move to functions
def get_lobular_time_signals(cb_data, fwd, estimate_data):
    lob_signs = []
    marks = np.unique(cb_data['parcellation']['surface'])[2:]
    cb_estimate = estimate_data[fwd['src'][0]['nuse']:, :]
    for d, mark in enumerate(marks):
        src_inds = np.where(np.isin(cb_data['parcellation']['surface'][cb_data['dw_data']['dense'][fwd['src'][1]['vertno']]], mark))[0]
        lob_signs.append(cb_estimate[src_inds, :])

    return lob_signs


def plot_lobular_tf(cb_data, fwd, estimate_data, ave):
    marks = np.unique(cb_data['parcellation']['surface'])[2:]
    fig, axs = plt.subplots(11, 3, dpi=300, figsize=(10,10), sharex=True, sharey=True)
    labels = ['lob I-III', 'lob I-III', 'lob IV', 'lob IV', 
              'lob V', 'lob V', 'lob VI', 'lob VI', 
              'lob VI', 'lob VII', 'crus I', 'crus II', 
              'lob VIIb', 'crus I', 'crus II', 'lob VIIb', 
              'lob VIII', 'lob VIIIa', 'lob VIIIb', 'lob VIIIa', 
              'lob VIIIb', 'lob IX', 'lob IX', 'lob IX', 
              'lob X', 'lob X', 'lob X']
    subplot_pos = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0], [2, 2], [3, 1], [3, 0],
                   [3, 2], [4, 1], [4, 0], [5, 0], [6, 0], [4, 2], [5, 2], [6, 2], 
                   [7, 1], [7, 0], [8, 0], [7, 2], [8, 2], [9, 1], [9, 0], [9, 2],
                   [10, 1], [10, 0], [10, 2]]
    axs[0, 0].set_title('left')
    axs[0, 1].set_title('vermis')
    axs[0, 2].set_title('right')
    lob_signs = get_lobular_time_signals(cb_data, fwd, estimate_data)

    fs = ave.info['sfreq']
    freq = np.linspace(1/ave.times[-1], 50, int(50/3))
    for d, mark in enumerate(marks):
        ave_sign = np.mean(np.abs(lob_signs[d]), axis=0)
        tf_morlet_wavelet(t=ave.times*1000, sig=ave_sign, freq=freq, fs=fs, ax=axs[subplot_pos[d][0], subplot_pos[d][1]],
                          threshold = 0, sat_threshold=.97);
        axs[subplot_pos[d][0], subplot_pos[d][1]].plot([0, 0],[freq[0], freq[-1]], linestyle='--', color='cyan', alpha=1)
        if subplot_pos[d][1] == 0:
            axs[subplot_pos[d][0], subplot_pos[d][1]].set_ylabel(labels[d]+'\n Freq.(Hz)', fontsize=7)
        if subplot_pos[d][0] == 10:
            axs[subplot_pos[d][0], subplot_pos[d][1]].set_xlabel('Time (ms)', fontsize=7)

    return fig, axs


def plot_mean_lobular_time_signals(cb_data, fwd, estimate_data, ave):
    marks = np.unique(cb_data['parcellation']['surface'])[2:]
    fig, axs = plt.subplots(11, 3, dpi=300, figsize=(10,10), sharex=True, sharey=True)
    alpha = 0.2
    labels = ['lob I-III', 'lob I-III', 'lob IV', 'lob IV', 
              'lob V', 'lob V', 'lob VI', 'lob VI', 
              'lob VI', 'lob VII', 'crus I', 'crus II', 
              'lob VIIb', 'crus I', 'crus II', 'lob VIIb', 
              'lob VIII', 'lob VIIIa', 'lob VIIIb', 'lob VIIIa', 
              'lob VIIIb', 'lob IX', 'lob IX', 'lob IX', 
              'lob X', 'lob X', 'lob X']
    subplot_pos = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0], [2, 2], [3, 1], [3, 0],
                   [3, 2], [4, 1], [4, 0], [5, 0], [6, 0], [4, 2], [5, 2], [6, 2], 
                   [7, 1], [7, 0], [8, 0], [7, 2], [8, 2], [9, 1], [9, 0], [9, 2],
                   [10, 1], [10, 0], [10, 2]]
    axs[0, 0].set_title('left')
    axs[0, 1].set_title('vermis')
    axs[0, 2].set_title('right')
    lob_signs = get_lobular_time_signals(cb_data, fwd, estimate_data)
    for d, mark in enumerate(marks):
        ave_sign = np.mean(np.abs(lob_signs[d]), axis=0)
        axs[subplot_pos[d][0], subplot_pos[d][1]].plot(ave.times*1000, ave_sign)
        if subplot_pos[d][0] == 10:
            axs[subplot_pos[d][0], subplot_pos[d][1]].set_xlabel('Time (ms)')
        axs[subplot_pos[d][0], subplot_pos[d][1]].axvline(x=0, linestyle='--', color='k', alpha=alpha)
        if subplot_pos[d][1] == 0:
            axs[subplot_pos[d][0], subplot_pos[d][1]].set_ylabel(labels[d])
    
    return fig, axs

#TBD correct here?
def tf_morlet_wavelet(t, sig, freq, fs, ax, threshold=0, sat_threshold=.98, w=6.):
    w=6. # cycles?
    widths = w*fs / (2*freq*np.pi)
    cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    vmax = (np.sort(np.abs(cwtm).flatten()))[int(sat_threshold*np.size(cwtm))]
    cwtm = np.abs(cwtm)
    tf_map = np.zeros(cwtm.shape)
    tf_map[np.where(cwtm > threshold*np.max(cwtm))] = cwtm[np.where(cwtm > threshold*np.max(cwtm))]
    ax.pcolormesh(t, freq, tf_map, cmap='hot_r', shading='gouraud', vmax=vmax)
#    ax.plot(tf_uncertainty, freq, 'r--', label='time-frequency Gabor uncertainty limit')
    ax.set_xlim([t[0], t[-1]])
#    ax.set_xlabel('time (s)')
#    ax.set_ylabel('frequency (Hz)')
#    plt.legend()
    return 


def plot_sagittal(vol, only_show_midline=False, **kwargs):
    sag_ind = kwargs.get('sag_ind')
    title = kwargs.get('title')
    rr = kwargs.get('rr')
    nn = kwargs.get('nn')
    tris = kwargs.get('tris')
    cmap = kwargs.get('cmap')
    linewidth = kwargs.get('linewidth')
    if type(cmap) == type(None):
        cmap = 'gray_r'
    if type(linewidth) == type(None):
        linewidth = 1.
    fig, ax = plt.subplots(3, 2)
    fig.suptitle(title)

    if type(sag_ind) == type(None):
        x_width = vol.shape[0]
        sag_ind = np.linspace(int(x_width*0.1), int(x_width*0.9), 6).astype(int)

    if only_show_midline:
        sag_ind = [sag_ind[3]]

    for c, slice_ind in enumerate(sag_ind):
        image = vol[slice_ind, :, :]
        plt.subplot(3, 2, c+1)
        plt.imshow(image, cmap=cmap)

        if not type(tris) == type(None):
            z_0 = slice_ind
            cart_ind = 0
            xy = [x for x in range(3) if not x==cart_ind] 
            intersecting_tris = []
            for tri in tris:
                rr_0 = rr[tri[0], :]
                rr_1 = rr[tri[1], :]
                rr_2 = rr[tri[2], :]
                if (np.array([np.sign((rr_0[cart_ind]-z_0)*(rr_1[cart_ind]-z_0)), 
                              np.sign((rr_0[cart_ind]-z_0)*(rr_2[cart_ind]-z_0)), 
                              np.sign((rr_1[cart_ind]-z_0)*(rr_2[cart_ind]-z_0))]) == -1).any():
                    intersecting_tris.append(tri)
            intersecting_tris=np.array(intersecting_tris)
            for int_tri in intersecting_tris:
                rr_0 = rr[int_tri[0], :]
                rr_1 = rr[int_tri[1], :]
                rr_2 = rr[int_tri[2], :]
                t_0 = (z_0 - rr_0[cart_ind])/(rr_1[cart_ind] - rr_0[cart_ind])
                t_1 = (z_0 - rr_0[cart_ind])/(rr_2[cart_ind] - rr_0[cart_ind])
                t_2 = (z_0 - rr_1[cart_ind])/(rr_2[cart_ind] - rr_1[cart_ind])
                xy_points = []
                if t_0 > 0 and t_0 < 1:
                    xy_points.append(t_0*rr_1[xy] + (1-t_0)*rr_0[xy])
                if t_1 > 0 and t_1 < 1:
                    xy_points.append(t_1*rr_2[xy] + (1-t_1)*rr_0[xy])
                if t_2 > 0 and t_2 < 1:
                    xy_points.append(t_2*rr_2[xy] + (1-t_2)*rr_1[xy])
                xy_points = np.array(xy_points)
                plt.plot(xy_points[:,1], xy_points[:,0], color='red', linewidth=linewidth)


        if not type(nn) == type(None):
            ptsp = np.where(np.abs(rr[:,0]-(slice_ind-0.5)) < 1.0)[0]
            x_tp = rr[ptsp,2]
            y_tp = rr[ptsp,1]
#            plt.scatter(x_tp, y_tp, color='r', s=0.1)
            plt.quiver(x_tp, y_tp, nn[ptsp,2], -nn[ptsp,1], scale=1, scale_units='inches')

    return fig, ax
