# vegetale
# Utilities used for the vegetale project
# 2020

__author__      = "Luis Salamanca"
__copyright__   = "Copyright 2020, ETH Zurich"


import os, sys
import numpy as np
import pickle as pk
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def group_datapoints_labels(data_dir, labels_dir, flag_rm = False,
                            str_new = '_all'):
    """
    Function for preprocessing the files and group them into a single dataframe
    We only convert and stack the similar ids
    New files stacking all these points are generated
    
    Parameters
    ----------
    data_dir : directory containing pkl files with independent samples
    labels_dir : directory containing pkl files with independent samples
    flag_rm : remove or not the original files

    Returns
    -------
    None.

    """
    
    all_filesid = np.array([o.split('.')[-2] for o in os.listdir(data_dir)])
    name_dataxyz = os.listdir(data_dir)[0].split('.')[0]

    all_labelsid = np.array([o.split('.')[-2] for o in os.listdir(labels_dir)])
    name_labels = os.listdir(labels_dir)[0].split('.')[0]
    
    common_ids = np.intersect1d(all_filesid, all_labelsid)
    
    df_data = []
    df_labels = []
    
    print('Start compiling data. Be patient!')
    n_percent = int(len(common_ids)/10)
    offset = 0
    for count, id_file in enumerate(common_ids):
        if count%n_percent == 0:
            print('{} percent, {} out of {}'.format(offset, count, len(common_ids)))
            offset += 10
        fnamed = '{}/{}.{}.pkl'.format(data_dir, name_dataxyz, id_file)
        df_data.append(pk.load(open(fnamed, 'rb')))
        fnamel = '{}/{}.{}.pkl'.format(labels_dir, name_labels, id_file)
        labels = pk.load(open(fnamel, 'rb'))
        df_labels.append(list(labels.values()))
        if flag_rm:
            os.remove(fnamed)
            os.remove(fnamel)

    df_data = pd.DataFrame(df_data)
    df_labels = pd.DataFrame(df_labels, columns=list(labels.keys()))
    
    print('Finished! Compiled {} out of {} (data) and {} (labels) total files'
          .format(len(common_ids), len(all_filesid), len(all_labelsid)))
    
    fname = '{}/{}{}.pkl'.format(data_dir, name_dataxyz, str_new)
    pk.dump(df_data, open(fname, 'wb'))
    fname = '{}/{}{}.pkl'.format(labels_dir, name_labels, str_new)
    pk.dump(df_labels, open(fname, 'wb'))
    
def wait_file(file_path):
    t_max = 3 # timeout
    t_inc = 0.01
    t = 0
    while os.path.exists(file_path)==False and t<t_max:
        time.sleep(t_inc)
        t += t_inc
    if t<t_max:
        st = os.stat(file_path)
        return st.st_size
    else:
        return -1
    
def plot_train_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

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
                     str_save = '', grid_supp = 11, flag_sdf = False):
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
    
    if flag_sdf:
        z_levels = np.zeros((num_samples, v))
        z_levels[:, 0] = bottom_top_heights[0]
        z_levels[:, -1] = bottom_top_heights[1]        
        z_levels[:, 1:-1] = w[:,-3:]
    else:
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
            if flag_sdf:
                [list_chal.append(list(map(float, w[ind, np.arange(grid_supp) + iv * grid_supp]))) for iv in range(v)]
                dict_aux['geo'] = {'sdf_radii' : list_chal}
                zl_aux = list(map(float, z_levels[ind, :]))
                dict_aux['geo']['z_levels'] = zl_aux
            else:
                #print(w_resh[ind, :])
                [list_chal.append([list(map(float, w_resh[ind, np.arange(3) + iv * 3])),
                                   list(map(float, w_resh[ind, 3 * v + np.arange(u) + iv * u]))]) for iv in range(v)]
                #dict_aux['geo'] = {'cpts_rrs' : list(w_resh[ind,:])}
                dict_aux['geo'] = {'cpts_rrs' : list_chal}
            
            if (x_in is not None) and (list_attrib is not None):
                dict_lab = dict()
                #print("LIST ATTR {}".format(list_attrib))
                for ind_lab, lab in enumerate(list_attrib):
                    #print("Ind lab {}".format(ind_lab))
                    #print("lab {}".format(lab))
                    try:
                        dict_lab[lab] = float(x_in[ind, ind_lab])
                    except:
                        dict_lab[lab] = 0.0
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

def ghlabels_to_script(x_dict, list_attributes = [], flag_all_df = False):
    """
    Convert the dictionary of labels we get from GH into an array, where
    the columns are ordered according to list_attributes

    Parameters
    ----------
    x_dict : dictionary obtained from GH
    list_attributes : list of labels to extract

    """

    #keys_dict = list(x_dict[0]['labels'].keys())
    #print(x_dict)
    #print(list(x_dict[0]))
    base_keys = list(x_dict[0]['labels'].keys())
    keys_dict = expand_keys_nestedlist(base_keys, x_dict[0]['labels'].values())
    x_lab = []
    ind_wrong_gh = []
    for i in range(len(x_dict)):
        if not len(x_dict[i]):
            x_lab.append(list(np.zeros(len(keys_dict))))
            ind_wrong_gh.append(i)
        else:
            vec_send = [x_dict[i]['labels'][att] for att in base_keys]
            #vec_el = nestedlist_to_list(list(x_dict[i]['labels'].values()))
            vec_el = nestedlist_to_list(vec_send)
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
                 flag_out_xyz_polar = 1,
                 flag_sdf = False):
    """
    Simply to convert from the structure in GH to the format
    we use in the autoencoder
    Then it is packed in a dictionary

    Parameters
    ----------
    x_dict : dictionary opened from GH file
    v, u : number of platforms and points
    flag_GH_xyz_polar : 0 for xyz, 1 for polar
    flag_out_xyz_polar : 0 for xyz, 1 for polar

    Returns
    -------
    x_resh : TYPE
        DESCRIPTION.
    bottom_top_heights : TYPE
        DESCRIPTION.

    """

    num_samples = len(x_dict)
    if flag_sdf:
        x = [[item for subl1 in o['geo']['sdf_radii'] for item in subl1] for o in x_dict]
        x = np.asarray(x)
        x_heights = [[item for item in o['geo']['z_levels'][1:-1]] for o in x_dict]
        x_heights = np.asarray(x_heights)
        x_resh = np.concatenate([x, x_heights], axis = 1)
        all_heights = x_dict[0]['geo']['z_levels']
        bottom_top_heights = [all_heights[0], all_heights[-1]]
    else:
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
                  list_plot_x = [[0,2],[1]], samp_to_plot = None):
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
    for isamp in vec_rand_samples[:1]:
        vec_heights = [1]
        [vec_heights.append(w_geom[isamp,-(v - 2 - o)]) for o in range(v - 2)]
        vec_heights.append(20)
        for iw in range(v):
            ind_use = np.arange(u * 2) + iw * u * 2
            points = w_geom[isamp,ind_use].reshape(u,-1)
            points = np.concatenate([points,points[0,:].reshape(1,2)], axis = 0)
            axs[iw].plot(points[:, 0], points[:, 1], c = colors[iw]) #, c = cmaplist[iw, :3].reshape(1,-1)
            axs[iw].set_title('Height of platform %s: %.2f' % (iw, vec_heights[iw]), fontsize=10)
            axs[v].plot(points[:, 0], points[:, 1], c = colors[iw]) #, c = cmaplist[iw, :3].reshape(1,-1)
            if points_inter_vec is not None:
                p_inter = points_inter_vec[iw]
                for pint in p_inter:
                    axs[iw].plot(pint[0], pint[1], marker = 'o', c = colors[iw])
        if x_vals is not None:
            x_val = x_vals[isamp,:]
            axs[v].set_title('%s: %.2f / %s: %.2f / %s: %.2f' % (list_att[0], x_val[0],
                                                                list_att[1], x_val[1],
                                                                list_att[2], x_val[2]), fontsize=10)
        else:
            axs[v].set_title('All platforms contours', fontsize=10)
        
        if len(list_plot_x) and x_vals is not None:
            for count, inds in enumerate(list_plot_x):
                if len(inds) == 2:
                    axs[v + count + 1].scatter(x_vals[::10,inds[0]], x_vals[::10,inds[1]])
                    axs[v + count + 1].scatter(x_vals[isamp,inds[0]], x_vals[isamp,inds[1]], marker = 'x',
                                           color = 'r')
                    axs[v + count + 1].set_title('%s and %s' % (list_att[inds[0]], list_att[inds[1]]))
                else:
                    axs[v + count + 1].hist(x_vals[:,inds[0]], 100, density=True)
                    axs[v + count + 1].plot([x_vals[isamp,inds[0]], x_vals[isamp,inds[0]]], [0, 0.1], color = 'r')
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


def line_intersection(line1, line2):
    # We don't consider here the case of two lines that just 
    xdiff = (line1[0,0] - line1[1,0], line2[0,0] - line2[1,0])
    ydiff = (line1[0,1] - line1[1,1], line2[0,1] - line2[1,1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the point belongs to both lines 
    all_cond = np.prod([(x <= np.max([line1[0,0],line1[1,0]])), (x >= np.min([line1[0,0],line1[1,0]])),
                        (x <= np.max([line2[0,0],line2[1,0]])), (x >= np.min([line2[0,0],line2[1,0]])),
                        (y <= np.max([line1[0,1],line1[1,1]])), (y >= np.min([line1[0,1],line1[1,1]])),
                        (y <= np.max([line2[0,1],line2[1,1]])), (y >= np.min([line2[0,1],line2[1,1]]))])
    if all_cond:
        return x, y
    else:
        return None
    
def line_int_nodet(l1, l2):
    # We don't consider here the case of two lines that just 
    t = ((l1[0,0] - l2[0,0])*(l2[0,1] - l2[1,1]) - (l1[0,1] - l2[0,1])*(l2[0,0] - l2[1,0]))/ \
        ((l1[0,0] - l1[1,0])*(l2[0,1] - l2[1,1]) - (l1[0,1] - l1[1,1])*(l2[0,0] - l2[1,0]))
    u = -((l1[0,0] - l1[1,0])*(l1[0,1] - l2[0,1]) - (l1[0,1] - l1[1,1])*(l1[0,0] - l2[0,0]))/ \
        ((l1[0,0] - l1[1,0])*(l2[0,1] - l2[1,1]) - (l1[0,1] - l1[1,1])*(l2[0,0] - l2[1,0])) 
    if (t > 0) and (t < 1) and (u > 0) and (u < 1):
        x = l1[0,0] + t*(l1[1,0] - l1[0,0])
        y = l1[0,1] + t*(l1[1,1] - l1[0,1])
        return x, y
    else:
        return None

def all_intersections(x, v, u, x_means = None, x_std = None):
    
    x = x.flatten()
    if x_means is not None:
        x = x * x_std.flatten() + x_means.flatten()
    n_inter_vec = []
    points_inter_vec = []
    for ix in range(v):
        ind_use = np.arange(u * 2) + ix * u * 2
        all_points = np.zeros((2, u * 2))
        all_points[1,:] = x[ind_use] 
        all_points[0,:2] = x[ind_use[-2:]] 
        all_points[0,2:] = x[ind_use[:-2]]
        all_points = np.swapaxes(np.swapaxes(all_points.reshape((-1,2,2)) , 0, 1), 1, 2)
        n_inter = 0
        points_inter = []
        for i in range(u):
            for j in range(i, u):
                p1 = all_points[:,:,i]
                p2 = all_points[:,:,j]
                if (p1[1,:] - p2[0,:]).sum() and (p1[0,:] - p2[1,:]).sum() and (i != j):
                    #res = line_intersection(p1, p2)
                    res = line_int_nodet(p1, p2)
                    #print(i,j)
                    if res is not None:
                        n_inter += 1
                        #print(ix, res, p1, p2)
                        points_inter.append(res)
                else:
                    stop = 1
                    #print('Nearby points',p1,p2)
        n_inter_vec.append(n_inter)
        points_inter_vec.append(points_inter)
    print(n_inter_vec)
    return n_inter_vec, points_inter_vec

def plot_diff_pred(x_in_ae, x_out_ae, x_out_gh, 
                   list_attrib_aux,
                   v = 5,
                   numb_p = 20,
                   ind_plot = [],
                   dim_to_plot = [['occlusion_rain','occlusion_sun'],['surface_areas_4', 'surface_areas_total']],
                   labels_dim_to_plot = [['Rain occlusion','Sun occlusion'],['Surface top platform', 'Total surface']],
                   flag_perSamp = False,
                   x_scatter = [],
                   flag_vert = False):
    
    from matplotlib.lines import Line2D
    
    list_attrib = copy.copy(list_attrib_aux)
    for iterm in ['outline_lengths_', 'surface_areas_']:
        if len(np.argwhere(np.asarray(list_attrib) == '{}0'.format(iterm))):
            ind_outline = []
            for iv in range(v):
                ind_outline.append(np.argwhere(np.asarray(list_attrib) == '{}{}'.format(iterm,iv)).flatten()[0])
                
            list_attrib.append('{}total'.format(iterm))
            ind_outline = np.asarray(ind_outline)
            
            x_in_ae = np.concatenate([x_in_ae, np.sum(x_in_ae[:,ind_outline], axis = 1).reshape(-1,1)], axis = 1)
            x_out_ae = np.concatenate([x_out_ae, np.sum(x_out_ae[:,ind_outline], axis = 1).reshape(-1,1)], axis = 1)
            x_out_gh = np.concatenate([x_out_gh, np.sum(x_out_gh[:,ind_outline], axis = 1).reshape(-1,1)], axis = 1)
            if len(x_scatter):
                x_scatter = np.concatenate([x_scatter, np.sum(x_scatter[:,ind_outline], axis = 1).reshape(-1,1)], axis = 1)
            
    #print(list_attrib, x_in_ae.shape)
    if not len(ind_plot):
        ind_plot = np.random.permutation(len(x_in_ae))[:numb_p]
    print(ind_plot)
    size_grid = int(np.ceil(len(list_attrib)/4))
    if flag_vert:
        fig, axs = plt.subplots(size_grid, 2, figsize=(20, 10 * size_grid))
    else:
        fig, axs = plt.subplots(2, size_grid, figsize=(10 * size_grid, 20))
    axs = np.array(axs).reshape(-1)
            
    ind_plots = []
    ind_1st_plot = [int(np.argwhere(np.asarray(list_attrib) == 'occlusion_rain')),
                    int(np.argwhere(np.asarray(list_attrib) == 'occlusion_sun'))]
    ind_plots.append(ind_1st_plot)
    
    ind_others_plot = np.setdiff1d(np.arange(len(list_attrib)), np.asarray(ind_1st_plot))            

    for i in range(int(np.ceil(len(ind_others_plot)/2))):
        if (i == (int(np.ceil(len(ind_others_plot)/2)) - 1)) and (len(ind_others_plot)%2 == 1):
            aux_inds = ind_others_plot[-1]
        else:
            aux_inds = ind_others_plot[np.arange(2) + i * 2]
        #print(aux_inds)
        ind_plots.append(list(aux_inds))
    
    all_max_hist = []
    if len(x_scatter):
        x_in_ae_aux = x_scatter
    else:
        x_in_ae_aux = x_in_ae
    for ii, inds_ in enumerate(ind_plots):
        #axs[ii].scatter(x_in_ae[:,inds_[0]], x_in_ae[:,inds_[1]], alpha = 0.5)
        #print(inds_, len(axs), ii ,inds_[0])
        if len(inds_) > 1:
            axs[ii].scatter(x_in_ae_aux[:,inds_[0]], x_in_ae_aux[:,inds_[1]], color = 'c', alpha = 0.3)
        else:
            axs[ii].hist(x_in_ae_aux[:,inds_[0]], 50, alpha = 0.3)
            aux1, _ = np.histogram(x_in_ae_aux[:,inds_[0]], 50, color = 'c', density = True)
            all_max_hist.append(np.max(aux1))
    
    for ii, inds_ in enumerate(ind_plots):
        for ind_ in ind_plot:
            
            if len(inds_) > 1:
                axs[ii].plot([x_in_ae[ind_,inds_[0]], x_out_ae[ind_,inds_[0]]],
                             [x_in_ae[ind_,inds_[1]], x_out_ae[ind_,inds_[1]]], 'g')
                axs[ii].plot([x_in_ae[ind_,inds_[0]], x_out_gh[ind_,inds_[0]]],
                             [x_in_ae[ind_,inds_[1]], x_out_gh[ind_,inds_[1]]], 'k')
                axs[ii].plot([x_out_ae[ind_,inds_[0]], x_out_gh[ind_,inds_[0]]],
                             [x_out_ae[ind_,inds_[1]], x_out_gh[ind_,inds_[1]]], '-r')  
                #axs[ii].text(x_out_ae[ind_,inds_[0]] + 2, x_out_ae[ind_,inds_[1]] + 2, str(ind_), fontsize = 24)
                #axs[ii].text(x_out_gh[ind_,inds_[0]] + 2, x_out_gh[ind_,inds_[1]] + 2, str(ind_), fontsize = 24)
                
                axs[ii].scatter(x_in_ae[ind_,inds_[0]], x_in_ae[ind_,inds_[1]], color = 'r')
                axs[ii].scatter(x_out_ae[ind_,inds_[0]], x_out_ae[ind_,inds_[1]], color = 'g', s = 82, marker = 's')
                axs[ii].scatter(x_out_gh[ind_,inds_[0]], x_out_gh[ind_,inds_[1]], color = 'k', s = 82, marker = 'd')        
                axs[ii].set_xlabel(list_attrib[inds_[0]], fontsize = 14)
                axs[ii].set_ylabel(list_attrib[inds_[1]], fontsize = 14)
            else:
                axs[ii].bar(x_in_ae[ind_,inds_[0]], all_max_hist[ii], color = 'r')
                axs[ii].bar(x_out_ae[ind_,inds_[0]], all_max_hist[ii], color = 'g', marker = 's')
                axs[ii].bar(x_out_gh[ind_,inds_[0]], all_max_hist[ii], color = 'k', marker = 'd')
                rand_num = np.random.random(1)[0]
                axs[ii].plot([x_in_ae[ind_,inds_[0]], rand_num * all_max_hist[ii]],
                             [x_out_ae[ind_,inds_[0]], rand_num * all_max_hist[ii]], 'g')
                axs[ii].plot([x_in_ae[ind_,inds_[0]], rand_num * all_max_hist[ii]],
                             [x_out_gh[ind_,inds_[0]], rand_num * all_max_hist[ii]], 'k')
                axs[ii].set_xlabel(list_attrib[inds_[0]], fontsize = 14)
            
        markers_el = [Line2D([0], [0], marker='o', color='r', markersize=8, label='Input AE'),
                      Line2D([0], [0], marker='s', color='g', markersize=8, label='Output AE'),
                      Line2D([0], [0], marker='d', color='k', markersize=8, label='Output GH')]
        if not ii:
            axs[ii].legend(handles = markers_el)
            
    if len(dim_to_plot):

        for ind_p, dim_vec in enumerate(dim_to_plot):
            inds_ = [np.argwhere(np.asarray(list_attrib) == dim_vec[0]).flatten(), 
                     np.argwhere(np.asarray(list_attrib) == dim_vec[1]).flatten()]
            plt.figure(figsize=(10, 10))
            plt.scatter(x_in_ae_aux[:,inds_[0]], x_in_ae_aux[:,inds_[1]], color = 'c', alpha = 0.3)
            for ind_ in ind_plot:
                plt.plot([x_in_ae[ind_,inds_[0]], x_out_ae[ind_,inds_[0]]],
                             [x_in_ae[ind_,inds_[1]], x_out_ae[ind_,inds_[1]]], 'g')
                plt.plot([x_in_ae[ind_,inds_[0]], x_out_gh[ind_,inds_[0]]],
                             [x_in_ae[ind_,inds_[1]], x_out_gh[ind_,inds_[1]]], 'k')
                plt.plot([x_out_ae[ind_,inds_[0]], x_out_gh[ind_,inds_[0]]],
                             [x_out_ae[ind_,inds_[1]], x_out_gh[ind_,inds_[1]]], '-r')  
                
                plt.scatter(x_in_ae[ind_,inds_[0]], x_in_ae[ind_,inds_[1]], color = 'r')
                plt.scatter(x_out_ae[ind_,inds_[0]], x_out_ae[ind_,inds_[1]], color = 'g', s = 82, marker = 's')
                plt.scatter(x_out_gh[ind_,inds_[0]], x_out_gh[ind_,inds_[1]], color = 'k', s = 82, marker = 'd')        
                plt.xlabel(labels_dim_to_plot[ind_p][0], fontsize = 14)
                plt.ylabel(labels_dim_to_plot[ind_p][1], fontsize = 14)
            
            
def extract_preds(ep_range, b_range,
                  list_attrib,
                  data_dir_loop = '',
                  names_files = []):
    
    def extract_x(dict_in, dict_out):
        x_in_ae = []
        x_out_gh = []        
        keys_dict_in = expand_keys_nestedlist(list(dict_in[0]['labels'].keys()), dict_in[0]['labels'].values())
        keys_dict_out = expand_keys_nestedlist(list(dict_out[0]['labels'].keys()), dict_out[0]['labels'].values())
        print(len(dict_in), len(dict_out))
        count = 0
        ind_valid_outgh = list()
        for dictin_, dictout_ in zip(dict_in, dict_out):
            if len(dictin_) and len(dictout_):
                nest_list = nestedlist_to_list(dictin_['labels'].values())
                df_list = pd.DataFrame(np.array(nest_list).reshape(1,-1), columns = keys_dict_in)
                x_in_ae_aux = [df_list[o] for o in list_attrib]
                nest_list = nestedlist_to_list(dictout_['labels'].values())
                if len(nest_list) == len(keys_dict_out):
                    #print(len(nest_list), len(keys_dict_out), nest_list, count)
                    df_list = pd.DataFrame(np.array(nest_list).reshape(1,-1), columns = keys_dict_out)
                    x_out_gh_aux = [df_list[o] for o in list_attrib]            
                    x_in_ae.append(x_in_ae_aux)
                    x_out_gh.append(x_out_gh_aux)
                    ind_valid_outgh.append(count)
                count += 1
            
        return x_in_ae, x_out_gh, ind_valid_outgh
    
    x_in_ae = []
    x_out_gh = []
    if not len(names_files):
        for ep in ep_range:
            for b in b_range:
                extra_str = 'E{}_b{}'.format(ep, b)
                f_name = os.path.join(data_dir_loop,'tmp_fromscript_{}.pkl'.format(extra_str))
                dict_in = pk.load(open(f_name,'rb'))
                f_name = os.path.join(data_dir_loop,'tmp_fromgh_{}.pkl'.format(extra_str))
                dict_out = pk.load(open(f_name,'rb'))
                x_in_ae_aux, x_out_gh_aux, ind_valid_outgh = extract_x(dict_in, dict_out)
                
                x_in_ae.append(x_in_ae_aux)
                x_out_gh.append(x_out_gh_aux)
    else:
        f_name = os.path.join(data_dir_loop, names_files[0])
        dict_in = pk.load(open(f_name,'rb'))
        f_name = os.path.join(data_dir_loop, names_files[1])
        dict_out = pk.load(open(f_name,'rb'))
        x_in_ae, x_out_gh, ind_valid_outgh = extract_x(dict_in, dict_out)


    return np.asarray(x_in_ae).reshape((-1, len(list_attrib))), np.asarray(x_out_gh).reshape((-1, len(list_attrib))), ind_valid_outgh

def plot_contours_sdf(w_geom, v, grid_supp, grid_pts_xy, bottom_top_heights = [4.5, 19], 
                      w_means = None, w_std = None, flag_save = False, count = 1, 
                      x_vals = None, list_att = ['OcRain', 'Surf', 'OcSun'],
                      list_plot_x = [[0,2],[1]], samp_to_plot = None):
    """
    Plotting the contours for one random sample, or one specified sample
    
    Parameters
    ----------
    w_geom : array of geometries, normalized or not, and in sdf format
    v, grid_supp : 
    grid_pts_xy : location of supports
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
    if x_vals is None:
        list_plot_x = []
    
    fig, axs = plt.subplots(2, int(np.ceil((len(list_plot_x) + v + 1)/2)), figsize=(15,7))
    # define the colormap
    colors = ['r','b','g','k','c','m','y']
    axs = axs.flat
    
    if samp_to_plot is None:
        vec_rand_samples = np.random.permutation(w_geom.shape[0])
    else:
        vec_rand_samples = [samp_to_plot]
        
    for isamp in vec_rand_samples[:1]:
        vec_heights = [bottom_top_heights[0]]
        [vec_heights.append(w_geom[isamp,-(v - 2 - o)]) for o in range(v - 2)]
        vec_heights.append(bottom_top_heights[1])
        if w_geom.shape[1] > v * grid_supp:
            w_geom = w_geom[:, :v * grid_supp]        
        w_geom_r = w_geom.reshape(v, grid_supp)
        axs[v].scatter(grid_pts_xy[:,0], grid_pts_xy[:,1], s = 80)
        for iw in range(v):
            points = w_geom_r[iw]
            
            axs[iw].scatter(grid_pts_xy[:,0], grid_pts_xy[:,1], s = 80)
            
            ind_diffzero = np.argwhere(points > 0)
            for indz in ind_diffzero:
                c = plt.Circle((grid_pts_xy[indz,0], grid_pts_xy[indz,1]), 
                               radius=points[indz], color = colors[iw], alpha = 0.4)
                axs[iw].add_patch(c)
                c = plt.Circle((grid_pts_xy[indz,0], grid_pts_xy[indz,1]), 
                               radius=points[indz], color = colors[iw], alpha = 0.2)
                axs[v].add_patch(c)
            axs[iw].set_title('Height of platform %s: %.2f' % (iw, vec_heights[iw]), fontsize=10)

        if x_vals is not None:
            x_val = x_vals[isamp,:]
            axs[v].set_title('%s: %.2f / %s: %.2f / %s: %.2f' % (list_att[0], x_val[0],
                                                                list_att[1], x_val[1],
                                                                list_att[2], x_val[2]), fontsize=10)
        else:
            axs[v].set_title('All platforms contours', fontsize=10)
        
        if len(list_plot_x) and x_vals is not None:
            for count, inds in enumerate(list_plot_x):
                if len(inds) == 2:
                    axs[v + count + 1].scatter(x_vals[::10,inds[0]], x_vals[::10,inds[1]])
                    axs[v + count + 1].scatter(x_vals[isamp,inds[0]], x_vals[isamp,inds[1]], marker = 'x',
                                           color = 'r')
                    axs[v + count + 1].set_title('%s and %s' % (list_att[inds[0]], list_att[inds[1]]))
                else:
                    axs[v + count + 1].hist(x_vals[:,inds[0]], 100, density=True)
                    axs[v + count + 1].plot([x_vals[isamp,inds[0]], x_vals[isamp,inds[0]]], [0, 0.1], color = 'r')
                    axs[v + count + 1].set_yticklabels([])
                    axs[v + count + 1].set_title('%s' % (list_att[inds[0]]))
        
        plt.show()
        if flag_save:
            plt.savefig('Contours_platforms_Sample{}.png'.format(isamp))
            plt.close()
        #time.sleep(20)
        #x = input('Continue plotting shapes? y/n')
        #fig.close()
        #if x == 'n':
        #    break
    
def generate_val_supp(adj_mat, v = 5, n_samp = 10, min_max_supp = [3,5]):
    
    num_s = np.round(np.random.uniform(min_max_supp[0] - 0.5, 
                                       min_max_supp[1] + 0.49, n_samp * v)).astype(int)
    
    grid_supp = adj_mat.shape[0]
    list_val_sup = []
    for i in range(grid_supp):
        list_val_sup.append(np.argwhere(adj_mat[i,:]))
    
    list_supp = []
    for ip in range(n_samp * v):
        aux_i = np.random.permutation(grid_supp)[0]
        list_supp_aux = [aux_i]
        not_end = 1
        while not_end:
            #print(np.setdiff1d(list_val_sup[aux_i], np.array(list_supp_aux)), list_val_sup[aux_i].flatten(), np.array(list_supp_aux))
            try:
                aux_i = np.random.permutation(np.setdiff1d(list_val_sup[aux_i].flatten(), np.array(list_supp_aux)))[0]
                list_supp_aux.append(aux_i)
                if len(list_supp_aux) == num_s[ip]:
                    not_end = 0
            except:
                aux_i = np.random.permutation(grid_supp)[0]
                list_supp_aux = [aux_i]
                
        list_supp.append(list_supp_aux)
        
    list_w_cat = np.zeros((n_samp * v, grid_supp)).astype(int)
    for ip in range(n_samp * v):
        list_w_cat[ip, np.array(list_supp[ip])] = 1
        
    return list_w_cat.reshape(-1, v * grid_supp), num_s

def generate_lut_const(adj_mat, min_max_supp = [3,5]):
    
    lut_const = []
    
    grid_supp = adj_mat.shape[0]
    list_val_sup = []
    for i in range(grid_supp):
        list_val_sup.append(np.argwhere(adj_mat[i,:]))
    
    for nsup in range(min_max_supp[0],min_max_supp[1] + 1):
        list_supp = np.arange(grid_supp).reshape(-1,1).astype(int)
        for ns in range(2, nsup + 1):
            
            list_supp_ext = []
            for const_s in range(list_supp.shape[0]):
                #print(list_val_sup[list_supp[const_s,-1]], list_supp[const_s,:])
                for const_c in range(list_supp.shape[1]):
                    vec_cat = np.setdiff1d(list_val_sup[int(list_supp[const_s,const_c])],list_supp[const_s,:])
                    if len(vec_cat):
                        list_supp_aux = np.zeros((len(vec_cat), ns))
                        list_supp_aux[:,:-1] = np.tile(list_supp[const_s,:], (len(vec_cat),1))
                        #print(vec_cat, vec_cat.shape)
                        list_supp_aux[:,-1] = vec_cat
                        if not len(list_supp_ext):
                            list_supp_ext = list_supp_aux
                        else:
                            list_supp_ext = np.concatenate([list_supp_ext, list_supp_aux], axis = 0)
            
            
            list_supp = np.sort(list_supp_ext, axis = 1).astype(int)
            list_supp_def = list()
            for row in list_supp:
                list_supp_def.append('_'.join(list(row.astype(str))))
            list_supp_def = np.unique(np.asarray(list_supp_def))
            
        lut_const.append(list_supp_def)
        
    tot_const = np.sum([len(o) for o in lut_const])
    full_list = np.concatenate(lut_const, axis = 0)
    lut_const_array = np.zeros((tot_const, grid_supp)).astype(int)
    for i in range(tot_const):
        ind_val =  np.asarray(full_list[i].split('_')).astype(int)
        lut_const_array[i, ind_val] = 1
            
    return lut_const, lut_const_array
    
    
def sample_Wcat_fromAE(file_varsamp, file_model, x_cond, 
                       data_dir = '', model_AEX_dec = None,
                       v = 5, n_samp_per = 10,
                       supp_grid = 11,
                        n_hidden_layers_dec = 2,
                        hidden_layer_dim = 40,
                        layer_size = 576):
    
    [mean_z_out_np, cov_z_out_np, list_const_arr] = pk.load(open(os.path.join(data_dir,file_varsamp),'rb'))
    list_const_arr_bin = array_to_bin(list_const_arr)
    mat_map = np.zeros(np.max(list_const_arr_bin) + 1).astype(int)
    for ind, indabs in enumerate(list_const_arr_bin):
        mat_map[indabs] = ind
    
    # Load the decoder
    import model_x_sdf_inWcat_onehot_condX as m_xfunc

    num_attr = x_cond.shape[1]
    
    if model_AEX_dec is None:
        model_AEX_dec = m_xfunc.Model(num_attr, v, 2, n_hidden_layers_dec, hidden_layer_dim = hidden_layer_dim,
                                      layer_size = layer_size, flag_enc_dec = 1, 
                                      list_const_array = list_const_arr)

        f_name = os.path.join(data_dir,file_model)

        model_prev = torch.load(f_name)
        model_dict = model_AEX_dec.state_dict()
        for key in model_prev.keys():
            if key in model_dict.keys():
                model_dict[key] = model_prev[key]

        model_AEX_dec.load_state_dict(model_dict)
    
    # Generate samples
    n_samp_generate = len(x_cond) * n_samp_per
    z_out_np_rand = np.random.multivariate_normal(mean_z_out_np, cov_z_out_np, n_samp_generate)
    
    # Repeat x_cond
    x_cond = np.repeat(x_cond, n_samp_per, axis = 0)
    
    # X and Z concatenate
    x_cond_z = np.concatenate([x_cond, z_out_np_rand], axis = 1)
    
    # Generate samples of w_cat
    x_cond_z = torch.tensor(x_cond_z, dtype = torch.float)
    w_out_logit, x = model_AEX_dec.forward(x_cond_z)
    #print(w_out_logit.size(), x_cond_z.size(), list_const_arr.shape[0])
    
    # Now obtain element with the maximum likelihood
    w_out_logit = w_out_logit.cpu().detach().numpy().reshape(-1, list_const_arr.shape[0])
    #print(w_out_logit.shape)
    ind_max = np.argmax(w_out_logit, axis = 1).astype(int)
    #print(ind_max)
    
    w_out_cat = []
    [w_out_cat.append(list_const_arr[o]) for o in ind_max]
    w_out_cat = np.asarray(w_out_cat).reshape(-1, v * supp_grid)
    
    return w_out_cat, ind_max, w_out_logit
    
def array_to_bin(arr):
    
    if np.ndim(arr) == 1:
        arr = arr.reshape(1,-1)
    ncols = arr.shape[1]
    arr_bin = np.sum(arr * (2**np.arange(ncols)).reshape(1,ncols), axis = 1)
    
    return arr_bin
    
def in_gh_to_cond_in(dictin, list_attr):
    x_cond = []
    attrib_cond = []
    
    dictin = {'labels': dictin}
    dict_list = [dictin]
    
    x = ghlabels_to_script(dict_list, flag_all_df = True)
    list_attr = np.concatenate([np.asarray(list_attr), 
                                ['outline_length_total', 'surface_area_total']])
    keys_use = np.intersect1d(list_attr, np.asarray(x.columns))
    print('Keys to use: ',keys_use)
    for d in range(x.shape[0]):
        x_cond_aux = []
        attrib_cond_aux = []
        for att in keys_use:
            if x.iloc[d][att] is not None:
                x_cond_aux.append(x.iloc[d][att])
                attrib_cond_aux.append(att)
        x_cond.append(x_cond_aux)
        attrib_cond.append(attrib_cond_aux)
    return x_cond, attrib_cond
        
    
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
    not_end = 1
    print('----- Waiting for a file')
    while not_end:
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
            print('---------- File opened!')
            os.remove(f_name_out)
            return dict_polar
        else:
            time.sleep(5)
            if (time.time() - in_time) > max_time:
                print('Max time reached!')
                return []
        
    