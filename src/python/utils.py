#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:19:51 2020

@author: luissalamanca
"""

"""
This scripts just contains utilities for converting the data formats, opening
grasshopper files, saving into compatible format for GH, etc. 
"""


import os, sys
import numpy as np
import pickle as pk
import pandas as pd
import time
import matplotlib.pyplot as plt


def reduce_polar(w, v, u, num_samples):
    """
    Reduce and adapt the polar coordinates with the format we need for the 
    pipeline: a center point, 8 radius for each platform, and the
    v - 2 heights we get from the vector before
    Hence, first u + 2 are the radius and the xy of center, and the last 
    v - 2 are the heoghts of the elements in the center
    """
    
    w_red = np.zeros((num_samples, v * (u + 2) + v - 2))
    for iv in range(v):
        w_red[:, np.arange(2) + iv * (2 + u)] = w[:, np.arange(2) + iv * 3] # Center
        w_red[:, np.arange(u) + iv * (2 + u) + 2] = w[:, 3 * v + np.arange(u) + iv * u] # Radius
    w_red[:,-(v - 2):] = w[:,5:(1 + 3*(v - 1)):3]
    bottom_top_heights = [w[0,2], w[0,2 + (v - 1)*3]]
    return w_red, bottom_top_heights

def polar_to_ghpolar(w, bottom_top_heights, v, u, flag_create_dict = True,
                     x_in = None, list_attrib = None, flag_save_files = False,
                     str_save = ''):
    """
    Simply to convert from the structure we use for the 
    autoencoder, to the format required by GH
    Then it is packed in a dictionary

    Parameters
    ----------
    w : geometries vector
    v, u : number of platforms and points
    flag_create_dict : create the dictionary with the format GH needs
    x_in : provided in case we also want to save the design attributes in the dict
    list_attrib : the design attributes to be used
    flag_save_files : allow saving the file while iteratively running GH in the loop
    str_save : the extra string used for saving the file

    """

    num_samples = w.shape[0]

    w_resh = np.zeros((num_samples,3 * v + u * v))
    for iv in range(v):
        w_resh[:, np.arange(2) + iv * 3] = w[:, np.arange(2) + iv * (2 + u)]  # Center
        w_resh[:, 3 * v + np.arange(u) + iv * u] = w[:, np.arange(u) + iv * (2 + u) + 2] # Radius
    w_resh[:,5:(1 + 3*(v - 1)):3] = w[:,-(v - 2):]
    w_resh[:,2] = bottom_top_heights[0]
    w_resh[:,2 + (v - 1)*3] = bottom_top_heights[1]

    if flag_create_dict:
        list_all_samples = []
        for ind in range(num_samples):
            dict_aux = dict()
            dict_aux = {'info' : {'generation' : ind + 1}}
            list_chal = []
            #print(w_resh[ind, :])
            [list_chal.append([list(map(float, w_resh[ind, np.arange(3) + iv * 3])),
                               list(map(float, w_resh[ind, 3 * v + np.arange(u) + iv * u]))]) for iv in range(v)]
            #dict_aux['geo'] = {'cpts_rrs' : list(w_resh[ind,:])}
            dict_aux['geo'] = {'cpts_rrs' : list_chal}
            
            if (x_in is not None) and (list_attrib is not None):
                #print("HERE")
                dict_lab = dict()
                for ind_lab, lab in enumerate(list_attrib):
                    #print("IND {}".format(ind))
                    #print("IND_LAB {}".format(ind_lab))
                    #print("LAB {}".format(lab))

                    dict_lab[lab] = float(x_in[ind, ind_lab])
                    #print(dict_lab[lab])
                dict_aux['labels'] = dict_lab
            
            if flag_save_files:
                dict_aux['flag_save'] = str_save
            else:
                dict_aux['flag_save'] = ''
            
            list_all_samples.append(dict_aux)
        return list_all_samples
    else:
        return w_resh
    
def nestedlist_to_list(nested_list):
    vec_el = []
    for i in nested_list:
        if type(i) == list:
            for j in i:
                vec_el.append(j)
        else:
            vec_el.append(i)
    return vec_el

def expand_keys_nestedlist(keys_dict, nested_list):
    vec_el = []
    for i, k in zip(nested_list, keys_dict):
        if type(i) == list:
            for c, j in enumerate(i):
                vec_el.append(k + '_' + str(c))
        else:
            vec_el.append(k)
    return vec_el


def ghlabels_to_script(x_dict, list_attributes, flag_all_df):
    """
    Convert the dictionary of labels we get from GH into an array, where
    the columns are ordered according to list_attributes

    Parameters
    ----------
    x_dict : dictionary obtained from GH
    list_attributes : list of labels to extract

    """

    #keys_dict = list(x_dict[0]['labels'].keys())
    keys_dict = expand_keys_nestedlist(list(x_dict[0]['labels'].keys()), x_dict[0]['labels'].values())
    x_lab = []
    ind_wrong_gh = []
    for i in range(len(x_dict)):
        if not len(x_dict[i]):
            x_lab.append(list(np.zeros(len(keys_dict))))
            ind_wrong_gh.append(i)
        else:
            vec_el = nestedlist_to_list(list(x_dict[i]['labels'].values()))
            #x_lab.append(list(x_dict[i]['labels'].values()))
            x_lab.append(vec_el)
    x_lab = pd.DataFrame(x_lab, columns = keys_dict)
    if flag_all_df:
        return x_lab
    
    x_lab_red = np.asarray(x_lab[list_attributes])
    if len(ind_wrong_gh):
        print('Wrongly computed elements in GH: ', ind_wrong_gh)
        ind_val = np.setdiff1d(np.arange(len(x_dict)),np.asarray(ind_wrong_gh))
        mean_val_fill = np.mean(x_lab_red[ind_val,:], axis = 0)
        for i in ind_wrong_gh:
            x_lab_red[i,:] = mean_val_fill
    return x_lab_red

def gh_to_script(x_dict, v, u, flag_GH_xyz_polar = 1,
                 flag_out_xyz_polar = 1):
    """
    Simply to convert from the structure we use for the 
    autoencoder, to the format required by GH
    Then it is packed in a dictionary


    Parameters
    ----------
    x_dict : dictionary opened from GH file
    v, u : number of platforms and points
    flag_GH_xyz_polar : 0 for xyz, 1 for polar
    flag_out_xyz_polar : 0 for xyz, 1 for polar

    Returns
    -------
    x_resh : design attributes reshaped
    bottom_top_heights : bottom and top height

    """

    num_samples = len(x_dict)
    
    if flag_GH_xyz_polar:
        x = [[item for subl1 in o['geo']['cpts_rrs'] for subl2 in subl1 for item in subl2] for o in x_dict]
        # We have to now organize it as it was before: first all centers,
        # and then all the points
        x = np.asarray(x)
        x_r = np.zeros(x.shape)
        list_ind = np.asarray([o for iv in range(v) for o in range(iv * (u + 3), iv * (u + 3) + 3)])
        x_r[:, :(3 * v)] = x[:, list_ind]
        list_ind = np.asarray([o for iv in range(v) for o in range(iv * (u + 3) + 3, iv * (u + 3) + 3 + u)])
        x_r[:, (3 * v):] = x[:, list_ind]
        x_resh, bottom_top_heights = reduce_polar(x_r, v, u, num_samples)
        if not flag_out_xyz_polar:
            x_resh = convert_toxyz(x_resh, v, u, num_samples)
    else:
        x = [[item for sublist in o['geo']['xyz'] for item in sublist] for o in x_dict]
        x = np.asarray(x)
        x_resh, bottom_top_heights = reduce_xyz(x, v, u, num_samples)
        if flag_out_xyz_polar:
            x_resh = convert_topolars(x_resh, v, u, num_samples)

    return x_resh, bottom_top_heights

def reduce_xyz(w, v, u, num_samples):
    """
    We just input some data, given as xyz points for the 8 platforms, and 
    the function simplifies it, preserving only xy, and the heights in between
    the min and the max. Hence, we move from v * u * 3 to v * u * 2 + (v - 2)

    Parameters
    ----------
    w : input data
    v, u : number of platforms and points
    num_samples : number of samples
    
    Returns
    -------
    The data flatten, with the coordinates xy as the first v * u * 2 elements,
    and the v - 2 heights at the end
    """
    
    if w.ndim != 4:
        try:
            w = w.reshape(num_samples,v,u,3)
        except:
            print('Dimensions where wrongly provided')
    
    w_red = w[:,:,:,:2].reshape(num_samples,-1)
    for i in range(1, v - 1):
        w_red = np.concatenate([w_red, w[:,i,0,2].reshape(-1,1)], axis = 1)
    bottom_top_heights = [w[0,0,0,2], w[0,v - 1,0,2]]
    
    return w_red, bottom_top_heights

def xyz_to_ghxyz(w, bottom_top_heights, v, u, flag_polar = False,
              flag_create_dict = True):
    # Simply to convert from the structure we use for the 
    # autoencoder, to the format required by GH
    # Then it is packed in a dictionary
    num_samples = w.shape[0]
    if flag_polar:
        w = convert_toxyz(w, v, u, num_samples)
    w_resh = np.zeros((num_samples,v,u,3))
    w_resh[:,:,:,:2] = w[:,:(v * u * 2)].reshape(num_samples,v,u,2)
    w_resh[:,0,:,2] = bottom_top_heights[0]
    w_resh[:,v - 1,:,2] = bottom_top_heights[1]
    for i in range(1, v - 1):
        w_resh[:,i,:,2] = np.tile(w[:,-(v - i - 1)].reshape((-1,1)), (1, u))

    w_resh = w_resh.reshape(num_samples,-1)
    if flag_create_dict:
        list_all_samples = []
        for ind in range(num_samples):
            dict_aux = dict()
            dict_aux = {'info' : {'generation' : ind}}
            #dict_aux['geo'] = {'xyz' : list(x_resh[ind,:])}
            list_chal = [[float(val) for val in w_resh[ind,range(u) + o * u]] for o in range(v)]
            dict_aux['geo'] = {'xyz' : list_chal}
            list_all_samples.append(dict_aux)
        return list_all_samples
    else:
        return w_resh
    
def convert_topolars(w, v, u, num_samples):
    """
    Receives the output vector provided by the function reduce_xyz
    u * 2 points (x and y) for each platform, and finally the v -2 heights
    We need to compute 8 radius for each platform, a center point, and the
    v - 2 heights we get from the vector before

    Parameters
    ----------
    w : geometries in xyz, after running function reduce_xyz
    v, u, num_samples : 

    Returns
    -------
    w_red_polar : geometries reduced ((2 + u) * v + (v-2) )

    """

    angles = np.arange(u) * 360/u
    ind_y = np.argwhere(angles == 0.).flatten() * 2 + 1
    ind_x = np.argwhere(angles == 90.).flatten() * 2
    # u radius per platform and 3 xyz for center, minus the 2 heights that are fixed
    w_red_polar = np.zeros((num_samples, v * (u + 3) - 2))
    for i in range(v):
        ind_use = np.arange(i * 2 * u, (i + 1) * 2 * u)
        #print(ind_use, ind_x, ind_y)
        w_center = w[:, ind_use[[ind_x[0] , ind_y[0]]]]
        #print(np.tile(x_center[:5,:], u))
        w_nosh = w[:, ind_use] - np.tile(w_center, u)        
        w_rad = np.sqrt(np.square(w_nosh[:,0::2]) + np.square(w_nosh[:,1::2]))
        #print(w[:1, ind_use], w_nosh[:1,:], w_center[:1,:], w_rad[:1,:])
        #print('###########')
        w_red_polar[:, np.arange(2) + i * (2 + u)] = w_center
        w_red_polar[:, np.arange(u) + i * (2 + u) + 2] = w_rad
    w_red_polar[:,-(v - 2):] = w[:,-(v - 2):]
    return w_red_polar
    
def convert_toxyz(w, v, u, num_samples):
    """
    Receives the output vector provided by the function convert_topolars
    u + 2 per platform, u radius and 2 for center (xy), and v - 2 heights

    Parameters
    ----------
    w : geometries in polar, after running function reduce_polar
    v, u, num_samples : 

    Returns
    -------
    w_red_polar : geometries reduced ((2 + u) * v + (v-2) )

    """    
    # Receives the output vector provided by the function convert_topolars
    # u + 2 per platform, u radius and 2 for center (xy), and v - 2 heights
    angles = np.arange(u) * 360/u
    w_red_xyz = np.zeros((num_samples, 2 * u * v + (v - 2)))
    for i in range(v):
        w_center = w[:, np.arange(2) + i * (u + 2)]
        w_rads = w[:, np.arange(u) + i * (u + 2) + 2]
        w_aux = np.zeros((num_samples, 2 * u))
        for j in range(u):
            w_aux[:, j * 2] = w_rads[:, j] * np.cos((2 * np.pi * angles[j])/360) + w_center[:,0]
            w_aux[:, j * 2 + 1] = w_rads[:, j] * np.sin((2 * np.pi * angles[j])/360) + w_center[:,1]
        w_red_xyz[:, np.arange(i * 2 * u, (i + 1) * 2 * u)] = w_aux
    w_red_xyz[:,-(v - 2):] = w[:,-(v - 2):]
    return w_red_xyz
        
def check_rows_nan(x):
    
    x = np.asanyarray(x).sum(axis = 1)
    ind_nan = np.argwhere(np.isnan(x)).flatten()
    ind_notnan = np.argwhere(np.isnan(x) == False).flatten()
    return ind_nan, ind_notnan

def plot_scatter_geometries(w_geom, v, u, w_means = None, w_std = None,
                            numb_samp = 1000):
    """
    Given a an array of geometries, it plots the blobs of points, to show 
    how they are distributed

    Parameters
    ----------
    w_geom : array of geometries, normalized or not, and in xyz or polar
    v, u : 
    w_means : normalization factor for the geometries
    w_std : scaling factor for the geometries
    numb_samp : number of samples to plot

    """
    # One scatter plot for each level
    # Unnormalize the geometries, if they are
    if np.ndim(w_geom) == 1:
        w_geom = w_geom.reshape((1,-1))    
    
    if w_means is not None:
        w_geom = w_geom * w_std + w_means
    
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = convert_toxyz(w_geom, v, u, w_geom.shape[0])
        
    # Vec of heights:
    vec_heights = [1]
    for iw in range(v - 2):
        vec_heights.append(np.mean(w_geom[:,-(v - 2 - iw)]))
    vec_heights.append(v)
    
    fig, axs = plt.subplots(2, int(np.ceil(v/2)), figsize=(15,7))
    # define the colormap
    cmap = plt.cm.gist_ncar
    # extract all colors from the .jet map
    cmaplist = np.array([cmap(i) for i in range(cmap.N)])
    cmaplist = cmaplist[np.round(np.linspace(0,250, num = u)).astype(int),:]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, u)
    axs = axs.flat
    vec_rand_samples = np.random.permutation(w_geom.shape[0])[:numb_samp]
    for ax, iw in zip(axs, range(v)):
        for ic in range(u):
            ind_use = np.arange(2) + iw * u * 2 +  ic * 2
            ax.scatter(w_geom[vec_rand_samples, ind_use[0]], w_geom[vec_rand_samples,ind_use[1]], c = cmaplist[ic, :3].reshape(1,-1), label = str(ic))
        ax.legend()
        ind_use = np.arange(u * 2) + iw * u * 2
        ind_center = ind_use[np.array([4,1])]
        ax.scatter(w_geom[vec_rand_samples, ind_center[0]], w_geom[vec_rand_samples,ind_center[1]], c = 'k', marker = 'x', label = 'Center')
        ax.set_title('Mean height of platform {}'.format(vec_heights[iw]), fontsize=10)
        
def plot_contours(w_geom, v, u, w_means = None, w_std = None,
                  points_inter_vec = None, flag_save = False,
                  count = 1, x_vals = None, list_att = ['OcRain', 'Surf', 'OcSun'],
                  list_plot_x = [[0,2],[1]], samp_to_plot = 0,grids=None,pole_ind=None):
    """
    Plotting the contours for one random sample, or one specified sample
    
    Parameters
    ----------
    w_geom : array of geometries, normalized or not, and in xyz or polar
    v, u : 
    w_means : normalization factor for the geometries
    w_std : scaling factor for the geometries
    points_inter_vec : in case we compute the intersecting points between
        edges
    flag_save: save figure of contours
    count: string for the name of the saved figure
    x_vals: design attributes, they have to be unnormalized, to plot their 
        distributions and the value chosen to plot the contours
    numb_samp : number of samples to plot
    list_attrib : the design attributes in x, in order
    list_plot_x : how to plot the dis
    """    
    # One scatter plot for each level
    # Unnormalize the geometries, if they are
    if np.ndim(w_geom) == 1:
        w_geom = w_geom.reshape((1,-1))
    if w_means is not None:
        w_geom = w_geom * w_std + w_means
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = convert_toxyz(w_geom, v, u, w_geom.shape[0])
    
    fig, axs = plt.subplots(2, int(np.ceil((len(list_plot_x) + v + 1)/2)), figsize=(15,7))
    # define the colormap
    colors = ['r','b','g','k','c','m','y']
    axs = axs.flat
    if samp_to_plot is None:
        vec_rand_samples = np.random.permutation(w_geom.shape[0])
    else:
        vec_rand_samples = [samp_to_plot]
    vec_heights = [1]
    [vec_heights.append(w_geom[samp_to_plot,-(v - 2 - o)]) for o in range(v - 2)]
    vec_heights.append(20)
    for iw in range(v):
        ind_use = np.arange(u * 2) + iw * u * 2
        points = w_geom[samp_to_plot,ind_use].reshape(u,-1)
        points = np.concatenate([points,points[0,:].reshape(1,2)], axis = 0)
        axs[iw].plot(points[:, 0], points[:, 1], c = colors[iw]) #, c = cmaplist[iw, :3].reshape(1,-1)
        axs[iw].set_title('Height of platform %s: %.2f' % (iw, vec_heights[iw]), fontsize=10)
        axs[v].plot(points[:, 0], points[:, 1], c = colors[iw]) #, c = cmaplist[iw, :3].reshape(1,-1)
        if points_inter_vec is not None:
            p_inter = points_inter_vec[iw]
            for pint in p_inter:
                axs[iw].plot(pint[0], pint[1], marker = 'o', c = colors[iw])
    if x_vals is not None:
        x_val = x_vals[samp_to_plot,:]
        axs[v].set_title('%s: %.2f / %s: %.2f / %s: %.2f' % (list_att[0], x_val[0],
                                                            list_att[1], x_val[1],
                                                            list_att[2], x_val[2]), fontsize=10)
    if grids is not None:
        if pole_ind is not None:
            ind = [i for i in range(11)]
            ind_not = np.setdiff1d(ind,pole_ind)
            ind1 = range(pole_ind[1])#[0,1,2]
            ind2 = range(pole_ind[1],pole_ind[2])#[3,4,5]
            ind3 = range(pole_ind[2],pole_ind[3])#[6,7,8]
            ind4 = range(pole_ind[3],pole_ind[4])#[9,10,11]
            ind5 = range(pole_ind[4],grids.shape[0])#[12,13,14]
            axs[v].scatter(grids[ind1,0],grids[ind1,1],c="cyan")
            axs[v].scatter(grids[ind2,0],grids[ind2,1],c="black")
            axs[v].scatter(grids[ind3,0],grids[ind3,1],c="green")
            axs[v].scatter(grids[ind4,0],grids[ind4,1],c="blue")
            axs[v].scatter(grids[ind5,0],grids[ind5,1],c="red")
        else:
            axs[v].scatter(grids[:,0],grids[:,1])
    else:
        axs[v].set_title('All platforms contours', fontsize=10)

    if len(list_plot_x) and x_vals is not None:
        for count, inds in enumerate(list_plot_x):
            if len(inds) == 2:
                axs[v + count + 1].scatter(x_vals[::10,inds[0]], x_vals[::10,inds[1]])
                axs[v + count + 1].scatter(x_vals[samp_to_plot,inds[0]], x_vals[samp_to_plot,inds[1]], marker = 'x',
                                       color = 'r')
                axs[v + count + 1].set_title('%s and %s' % (list_att[inds[0]], list_att[inds[1]]))
            else:
                axs[v + count + 1].hist(x_vals[:,inds[0]], 100, density=True)
                axs[v + count + 1].plot([x_vals[samp_to_plot,inds[0]], x_vals[samp_to_plot,inds[0]]], [0, 0.1], color = 'r')
                axs[v + count + 1].set_yticklabels([])
                axs[v + count + 1].set_title('%s' % (list_att[inds[0]]))
    plt.show()
    if flag_save:
        plt.savefig('Contours_platforms_{}.png'.format(count))
        plt.close()
    #time.sleep(20)
    #x = input('Continue plotting shapes? y/n')
    #fig.close()
    #if x == 'n':
    #    break
    
def wait_gh_x(f_name_out, max_time):
    """
    Script that wait for a file to be created, checking also that the size remains
    stable to avoid reading a corrupted file. 

    Parameters
    ----------
    f_name_out : name of the file to open
    max_time : maximum time before existing the loop

    Returns
    -------
    Content of the file open

    """
    # Wait for file and open the content
    in_time = time.time()
    # We first convert them to the format required by GH
    iter = 0
    not_end = 1
    while not_end:
        iter += 1
        if os.path.exists(f_name_out):
            check_size = 1
            prev_size = -1
            while check_size:
                if prev_size == os.path.getsize(f_name_out):
                    check_size = 0
                else:
                    prev_size = os.path.getsize(f_name_out)
                    time.sleep(2)
            dict_polar = pk.load(open(f_name_out,'rb'))
            #print("GOT IT")
            #print(dict_polar[:4])
            os.remove(f_name_out)
            return dict_polar
        else:
            time.sleep(5)
            if (time.time() - in_time) > max_time:
                print('Max time reached!')
                return []