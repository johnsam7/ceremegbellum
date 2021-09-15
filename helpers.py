import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import nibabel as nib
import pickle
import os
import evaler
from scipy import signal

def change_labels(vol, old_labels, new_labels):
    new_vol = vol.copy()
    for c, old_label in enumerate(old_labels):
        mask_inds = np.where(vol == old_label)
        new_vol[mask_inds[0], mask_inds[1], mask_inds[2]] = new_labels[c]
    return new_vol

def switch_atlas_labels(labels, cb_data):
    for key in list(cb_data['parcellation']['dictionary'].keys())[1:len(cb_data['parcellation']['dictionary'].keys())]:
        old_ind = cb_data['parcellation']['dictionary'][key][0]
        new_ind = cb_data['parcellation']['dictionary'][key][1]
        labels[np.where(labels == old_ind)] = new_ind
    return labels

def rotation(angles, rr):
    """
    Rotates points rr around x-axis angles[0], y-axis angles[1] and z-axis angles[2]
    around its center of gravity.
    """
    a, b, c = angles
    rot_mat = np.array([[np.cos(b)*np.cos(c), -np.cos(b)*np.sin(c), np.sin(b)], 
                   [np.sin(a)*np.sin(b)*np.cos(c) + np.cos(a)*np.sin(c), 
                    -np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c),
                    -np.sin(a)*np.cos(b)], 
                   [ -np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c),
                    np.cos(a)*np.sin(b)*np.sin(c) + np.sin(a)*np.cos(c),
                    np.cos(a)*np.cos(b)]])
    rr_center = np.mean(rr, axis=0)
    rr_n = rr - rr_center
    rr_n = np.dot(rot_mat, rr_n.T).T
    rr_n = rr_n + rr_center
    return rr_n

def translation(r_0, rr):
    """
    Translates points rr by r_0
    """
    return rr + r_0

def scale(c, rr):
    """
    Scales points rr by c
    """
    return c*rr

def affine_transform(c, r_0, angles, rr):
    """
    Performs an affine transformation by rotation, translation and scaling.
    """
 #   r_center = np.mean(rr, axis=0)
 #   rr = rr - r_center
    rr = rotation(angles, rr)
    rr = translation(r_0, rr)
    rr = scale(c, rr)
    return rr

def solid_body_transform(r_0, angles, rr):
    """
    Performs a solid body transformation by rotation and translation.
    """
 #   r_center = np.mean(rr, axis=0)
 #   rr = rr - r_center
    rr = rotation(angles, rr)
    rr = translation(r_0, rr)
    return rr

def cost_boundbox(para, rr, rr_0):
    c = para[0]
    r_0 = para[1:4]
    angles = para[4:7]
    
    rr_n = affine_transform(c, r_0, angles, rr)

    r_abs = np.array([np.min(rr_n[:,0]), np.max(rr_n[:,0]), np.min(rr_n[:,1]),
             np.max(rr_n[:,1]), np.min(rr_n[:,2]), np.max(rr_n[:,2])])

    r_0_abs = np.array([np.min(rr_0[:,0]), np.max(rr_0[:,0]), np.min(rr_0[:,1]),
               np.max(rr_0[:,1]), np.min(rr_0[:,2]), np.max(rr_0[:,2])])
    
    return np.linalg.norm(r_abs-r_0_abs) 

def cost_contrast(para, weights, high_vols_cx, high_vols_wm, subj_vols):
    c = para[0]
    r_0 = para[1:4]
    angles = para[4:7]
    rr_cx_n = np.rint(affine_transform(c, r_0, angles, high_vols_cx)).astype(int)
    rr_wm_n = np.rint(affine_transform(c, r_0, angles, high_vols_wm)).astype(int)
    if np.max(rr_cx_n) > 255:
        return np.inf
    vol_data_n = np.zeros((256, 256, 256, 3))
    vol_data_n[:, :, :, :] = np.array([0, 0, 1])
    vol_data_n[rr_cx_n[:, 0], rr_cx_n[:, 1], rr_cx_n[:, 2], :] = np.array([1, 0, 0])
    vol_data_n[rr_wm_n[:, 0], rr_wm_n[:, 1], rr_wm_n[:, 2], :] = np.array([0, 1, 0])
    cost = weights[0]*np.linalg.norm(vol_data_n[:, :, :, 0] - subj_vols[:, :, :, 0]) + \
           weights[1]*np.linalg.norm(vol_data_n[:, :, :, 1] - subj_vols[:, :, :, 1]) + \
           weights[2]*np.linalg.norm(vol_data_n[:, :, :, 2] - subj_vols[:, :, :, 2])
    return cost

def bound_box_fit(high_res_verts, subj_cx_verts):
    s_0 = (np.max(high_res_verts[0,:]) - np.min(high_res_verts[0,:])) / (np.max(subj_cx_verts[0,:]) - np.min(subj_cx_verts[0,:]))
    r_0 = np.mean(subj_cx_verts, axis=0) - np.mean(high_res_verts, axis=0)
    
    res = minimize(cost_boundbox, np.array([s_0, r_0[0], r_0[1], r_0[2], 0, 0, 0]), 
                   (high_res_verts, subj_cx_verts))
    para = res['x']
    
    rr_fitted = affine_transform(para[0], para[1:4], para[4:7], high_res_verts)

    return rr_fitted

def find_connected_regions(vol, print_progress=True):
    """
    Finds connected regions in vol labeled by integers except for 0.
    """
    f_vox2int = {}
    voxels_removed = np.array([[]]).reshape((0,3))
    delta = np.array([[[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)] for z in np.arange(-1, 2)]).reshape(27,3)
    labels = list(np.unique(vol))
    labels.remove(0)
    labels2regions = {}

#    for c, vox in enumerate(non_zero_voxs):
#        ind = vol2int(vox, max_vol=np.max(vol.shape))
#        f_vox2int.update({tuple(vox) : ind})

    for val in labels:
        voxels_in_label = np.array(np.where(vol==val)).T

        f_vox2region = {}
        for c, vox in enumerate(voxels_in_label):
            f_vox2region.update({tuple(vox) : 0}) # 0 means "unassigned"
        
        regions = []
        while 0 in list(f_vox2region.values()):
            unassigned_voxels = np.array(list(f_vox2region.keys()))[np.where(np.array(list(f_vox2region.values()))==0)[0]]
            f_vox2region.update({tuple(unassigned_voxels[0]) : 1})
            front_line_vols = unassigned_voxels[0].reshape((1,3))
            saved_vols = [tuple(unassigned_voxels[0])]
            while len(front_line_vols) > 0:
                for vox in front_line_vols:
                    neighbors = vox + delta
                    for neighbor in neighbors:
        #                if not vol2int(neighbor, max_vol) in black_vols:
                        if tuple(neighbor) in list(f_vox2region.keys()):
                            if f_vox2region[tuple(neighbor)] == 0:
                                front_line_vols = np.vstack((front_line_vols, neighbor))
                                saved_vols.append(tuple(neighbor))
                                f_vox2region.update({tuple(neighbor) : 1})
            #                    black_vols.append(vol2int(neighbor, max_vol))
    #                            black_vols.append(f_vox2int[tuple(neighbor)])
                    front_line_vols = front_line_vols[1: front_line_vols.shape[0], :]
                if print_progress:
                    print(len(front_line_vols))
            regions.append(saved_vols)
        labels2regions.update({val : regions})
        if print_progress:
            print('Done with label '+str(val))
        
    return labels2regions
    
def print_ply(rr, tris, ply_fname, nn=None):
    my_list = []

    if type(nn) == np.ndarray:
        for x, y in zip(rr, nn):
            temp_tuple = (x[0], x[1], x[2], y[0], y[1], y[2])            
            my_list.append(temp_tuple)
        vertices = np.array(my_list, dtype=[('x', 'float64'), ('y', 'float64'), 
                                            ('z', 'float64'), ('nx', 'float64'),
                                            ('ny', 'float64'), ('nz', 'float64')])
    else:
        for x in rr:
            temp_tuple = (x[0], x[1], x[2])            
            my_list.append(temp_tuple)
        vertices = np.array(my_list, dtype=[('x', 'float64'), ('y', 'float64'), 
                                            ('z', 'float64')])
    
    my_list = []
    for x in tris:
        data_tup = ([x[0], x[1], x[2]],)
        my_list.append(data_tup)   
    
    faces = np.array(my_list, dtype=[('vertex_indices', 'int32', (3,)),])
    
    # Print ply
    el_vert = PlyElement.describe(vertices,'vertex')
    el_face = PlyElement.describe(faces,'face')
    PlyData([el_vert, el_face], text=True).write(ply_fname)
    print('surface data saved to ' + ply_fname)
    return

