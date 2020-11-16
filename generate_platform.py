import numpy as np
import os
import pickle as pk

from src.python import utils_vegetale as ut
import random
import math
from scipy.stats import norm, dirichlet, multivariate_normal
import matplotlib.pyplot as plt
from itertools import chain
import scipy
from sympy.utilities.iterables import multiset_permutations




"""
From total area and perimeter to 5 platform geometries

Generative Process for Platforms given a total surface (TS) and total perimeter (TP).

0 Describe each surface in your data base by R = Perimeter^2 / Area.

For any new structure:

1 Generate from a s = Dirichlet(10,10,10,10,10) a sample that has the surface for each one of the platforms. The area for each platform will be S_i = TS* s_i for i= 0, …, 4.

2 Compute D_TP = TP - \sum 2 \sqrt {\pi *S_i}. If every platform is a circle, D_TP measures the residual perimeter.

3 Generate one sample p = Dirichlet(10,10,10,10,10), which computes the extra perimeter per platform. Set P_i = 2 \sqrt {\pi *S_i} + p_i * D_TP.

4 Compute R_i = P_i^2 / S_i.

5 Find in the database of platforms those closes to R_i.

"""

curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/'  # + f_experiment
results_dir = curr_dir_repo + '/results/'  # + f_experiment

list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']



def load_grids(grid_name_string='grid_points_coord.pkl'):
    gridfile = os.path.join(dataset_dir + "/grid/", grid_name_string)
    grids = pk.load(open(gridfile, 'rb'))
    return grids



def load_raw_data(list_attrib, tot_area_perim_attr,u = 8, v = 5, flag_polar=True,
                  fname_string='data_labels_SDF_200830_470K_cl_flags_occl.pkl',
                  flag_sdf=True):
    print("Loading raw data...")


    fname = os.path.join(dataset_dir, fname_string)
    list_all_dicts = pk.load(open(fname, 'rb'))


    x_train = ut.ghlabels_to_script(list_all_dicts, flag_all_df=True)
    total_area_perim = np.asarray(x_train[tot_area_perim_attr])

    x_train = x_train[list_attrib]

    flag_GH_xyz_polar = 0
    flag_out_xyz_polar = 0
    if flag_polar:
        flag_GH_xyz_polar = 1
        flag_out_xyz_polar = 1

    w_train_red, bottom_top_heights = ut.gh_to_script(list_all_dicts, v, u, flag_sdf = flag_sdf)

    w_nans, ind_not_nanw = ut.check_rows_nan(w_train_red)
    """
    if len(w_nans):
        ind_not_nan = ind_not_nanw
        print('Reducing samples from {} to {}'.format(len(w_train_red), len(ind_not_nan)))
        w_train_red = w_train_red[ind_not_nan, :]
    else:
        print('No nans! ¯\_(ツ)_/¯')
    """

    return w_train_red, x_train, total_area_perim, bottom_top_heights

"""
def find_nearest(array, values):
    array = np.asarray(array)
    indices = []
    platforms = []
    tmp_array = array
    for v in values:
        idx_min = (tmp_array-v).argmin()
        val_min = tmp_array[idx_min]
        tmp_array = np.delete(tmp_array,np.argwhere(tmp_array==val_min))
        idx = np.argwhere(array==val_min)[0]
        picked = np.random.choice(idx,size=1)
        indices.append(picked)
        platforms.append(array[picked])


    return platforms, indices

"""
def find_nearest(array, value):
    indices = []
    platforms = []
    idx_min = np.abs(array-value).argmin()
    val_min = array[idx_min]
    indices.append(idx_min)
    platforms.append(val_min)
    return val_min, idx_min


    return platforms, indices
def plot_platforms(w_geom, v=5, u=8, points_inter_vec=None, flag_save=False,
                  count=1, list_plot_x=[[0, 2], [1]]):
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
        w_geom = w_geom.reshape((1, -1))
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_geom, v, u, w_geom.shape[0])

    fig, axs = plt.subplots(2, int(np.ceil((len(list_plot_x) + v + 1) / 2)), figsize=(15, 7))
    # define the colormap
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
    axs = axs.flat

    vec_heights = [1]
    [vec_heights.append(w_geom[0, -(v - 2 - o)]) for o in range(v - 2)]
    vec_heights.append(20)
    for iw in range(v):
        ind_use = np.arange(u * 2) + iw * u * 2
        points = w_geom[0, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        axs[iw].plot(points[:, 0], points[:, 1], c=colors[iw])  # , c = cmaplist[iw, :3].reshape(1,-1)
        axs[iw].set_title('Height of platform %s: %.2f' % (iw, vec_heights[iw]), fontsize=10)
        axs[v].plot(points[:, 0], points[:, 1], c=colors[iw])  # , c = cmaplist[iw, :3].reshape(1,-1)
        if points_inter_vec is not None:
            p_inter = points_inter_vec[iw]
            for pint in p_inter:
                axs[iw].plot(pint[0], pint[1], marker='o', c=colors[iw])



    plt.show()
    if flag_save:
        plt.savefig('Contours_platforms_{}.png'.format(count))
        plt.close()

def possible_permutations(platform, all_perm):
    # shift platform
    # check if valid

    # make first a grid (all pairwise distances) of current position

    #indecies = np.where(platform!= 0)
    vals = platform[platform!=0.0]
    all_vals = []
    for i in range(vals.shape[0]):
        all_vals.append(np.roll(vals,i))
    #support_loc = supports[indecies,:]
    #distances = np.linalg.norm(support_loc[:, None, :] - support_loc[None, :, :], axis=-1)

    # now find all possible configurations, that keep these distances

    # how tf do we do that??

    # first all possible distances

    #all_distances = np.linalg.norm(supports[:, None, :] - supports[None, :, :], axis=-1)

    all_ind = np.where(platform > 0.0, 1, 0)

    possible_inds = []
    n = 0

    m = 0
    #all_perm = all_perm[:,:-3]
    all_perm_list = all_perm.tolist()
    #all_perm = all_perm.reshape(all_perm.shape[0]*5,11)
    for p in multiset_permutations(all_ind):
        #print(p)
        #print(p.shape)
        #tmp = supports[p, :]
        for pm in all_perm_list:
            #print(pm)
            #print(pm.shape)
            if p == pm:
                possible_inds.append(p)
                break
    configurations = np.empty((1,11))
    for positions in possible_inds:
        #print(positions)
        inds = np.argwhere(np.array(positions)>0.0).flatten()
        for v in all_vals:
            tmp = np.array(positions).astype(float)
            np.place(tmp,tmp>0.0,v)
            configurations = np.vstack((configurations, tmp))

    configurations = np.unique(configurations, axis=0)
    configurations = configurations.tolist()

        #tmp_distances = np.linalg.norm(tmp[:, None, :] - tmp[None, :, :], axis=-1)

        #is_eq = np.array_equal(distances,tmp_distances)
        #if is_eq:
        #   possible_inds.append(tmp)

    return possible_inds, configurations

def generate_xyz(fname, w_train_red, x_train,total_area_perim,desired_perims = None, desired_areas = None):
    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    tot_area_perim_attr = ['surface_area_total', 'outline_length_total']

    #w_train_red, x_train, total_area_perim = load_raw_data(list_attrib, tot_area_perim_attr)
    #w_train_red, x_train, total_area_perim, _ = load_raw_data(list_attrib, tot_area_perim_attr,
    #                                                          fname_string=fname, flag_sdf=False)

    perims = np.asarray(x_train[['outline_lengths_0', 'outline_lengths_1',
                                 'outline_lengths_2', 'outline_lengths_3', 'outline_lengths_4']])
    areas = np.asarray(x_train[['surface_areas_0', 'surface_areas_1',
                                'surface_areas_2', 'surface_areas_3', 'surface_areas_4']])

    perims = perims.flatten()
    areas = areas.flatten()

    mu_p, std_p = norm.fit(total_area_perim[:, 1])
    # mean_p = np.mean(perims, axis=0)
    # cov_p = np.cov(perims, rowvar=0)

    mu_a, std_a = norm.fit(total_area_perim[:, 0])
    # mean_a = np.mean(areas, axis=0)
    # cov_a = np.cov(areas, rowvar=0)
    N = 1

    #desired_perims = np.array([260.94084443])  # np.random.normal(loc=mu_p, scale=std_p, size=N)
    desired_perims = np.array([desired_perims])  # np.random.normal(loc=mu_p, scale=std_p, size=N)

    #desired_areas = np.array([123.63451648]) # np.random.normal(loc=mu_a, scale=std_a, size=N)
    desired_areas = np.array([desired_areas]) # np.random.normal(loc=mu_a, scale=std_a, size=N)

    # desired_perims = np.random.shuffle(desired_perims)
    # desired_areas = np.random.shuffle(desired_areas)
    # desired_perims = desired_perims[0]
    # desired_areas = desired_areas[0]
    # desired_areas = np.random.choice(total_area_perim[:,0],size=1)
    # desired_perims = np.random.choice(total_area_perim[:,1],size=1)
    print(desired_areas)
    print(desired_perims)

    # tot = np.empty((N, 2))
    # tot[:, 0] = 38
    # tot[:, 1] = 129

    R = perims ** 2 / areas
    alpha = 10

    s = dirichlet.rvs([alpha, alpha, alpha, alpha, alpha], size=1, random_state=1)
    S = desired_areas * s
    S = S[0]
    ss = 0
    for s_i in S:
        ss += 2 * np.sqrt(np.pi * s_i)

    d_tp = desired_perims - ss

    p = dirichlet.rvs([alpha, alpha, alpha, alpha, alpha], size=1, random_state=1)
    # Set P_i = 2 \sqrt {\pi *S_i} + p_i * D_TP.
    P = 2 * np.sqrt(np.pi * S) + p * d_tp
    P = P[0]
    # 4 Compute R_i = P_i^2 / S_i.
    R_des = P ** 2 / S
    R_des = R_des.reshape(5)
    print(R.shape)
    print(R_des.shape)
    index = np.empty((5))
    for i in range(5):
        _, ind = find_nearest(R, R_des[i])
        index[i] = ind
    index = index.astype(int)
    # index = np.array(index)
    # index = index.flatten()
    # print(index)
    # print(nearest_in_dataset)

    xy_ind = list(chain.from_iterable((i, i + 1) for i in range(0, 50, 10)))
    xy_ind.append(50)
    xy_ind.append(51)
    xy_ind.append(52)
    # print(xy_ind)
    all_ind = np.array(range(53))
    take_ind = np.setdiff1d(all_ind, xy_ind)
    # print(len(take_ind))
    w_xyrad = w_train_red[:, take_ind]
    # print(w_xyrad.shape[0]*5)
    print(index)
    single_platforms = np.empty((R.shape[0], 8))
    for row in range(w_xyrad.shape[0]):
        single_platforms[5 * row, :] = w_xyrad[row, :8]
        single_platforms[5 * row + 1, :] = w_xyrad[row, 8:16]
        single_platforms[5 * row + 2, :] = w_xyrad[row, 16:24]
        single_platforms[5 * row + 3, :] = w_xyrad[row, 24:32]
        single_platforms[5 * row + 4, :] = w_xyrad[row, 32:40]

    # w_xyrad = w_xyrad.flatten()
    # print(w_xyrad.shape)
    # print(R.shape)
    # %%

    # w_xyrad = np.reshape(w_xyrad, (-1, 10))
    # platforms = np.empty((5,w_xyrad.shape[1]+3))
    platforms = np.take(single_platforms, index, axis=0)
    platforms_to_plot = np.empty((1, 53))
    for ind, val in zip(take_ind, platforms.flatten()):
        platforms_to_plot[:, ind] = val
    # platforms_to_plot[:,take_ind] = platforms.flatten()

    mean_xy = np.mean(w_train_red[:, xy_ind], axis=0)
    cov_xyz = np.cov(w_train_red[:, xy_ind], rowvar=0)
    platforms_to_plot[:, xy_ind] = np.random.multivariate_normal(mean=mean_xy, cov=cov_xyz)

    # plot_platforms(w_train_red[0,:])

    return platforms, desired_areas, desired_perims

def generate_sdf(fname, w_train_red, x_train, total_area_perim, desired_perims = None, desired_areas = None):
    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    tot_area_perim_attr = ['surface_area_total', 'outline_length_total']

    #w_train_red, x_train, total_area_perim,_ = load_raw_data(list_attrib, tot_area_perim_attr,
    #                                                         fname_string=fname, flag_sdf=True)
    #print(total_area_perim.shape)
    #print(w_train_red.shape)

    perims = np.asarray(x_train[['outline_lengths_0', 'outline_lengths_1',
                                 'outline_lengths_2', 'outline_lengths_3', 'outline_lengths_4']])
    areas = np.asarray(x_train[['surface_areas_0', 'surface_areas_1',
                                'surface_areas_2', 'surface_areas_3', 'surface_areas_4']])

    perims = perims.flatten()
    #print("perim {}".format(perims.shape))
    areas = areas.flatten()


    mu_p, std_p = norm.fit(total_area_perim[:, 1])
    #mean_p = np.mean(perims, axis=0)
    #cov_p = np.cov(perims, rowvar=0)

    mu_a, std_a = norm.fit(total_area_perim[:, 0])
    #mean_a = np.mean(areas, axis=0)
    #cov_a = np.cov(areas, rowvar=0)
    N = 1

    #desired_perims = 260.94084443#np.random.normal(loc=mu_p, scale=std_p, size=N)

    #desired_areas = 123.63451648#np.random.normal(loc=mu_a, scale=std_a, size=N)

    # desired_perims = np.array([260.94084443])  # np.random.normal(loc=mu_p, scale=std_p, size=N)
    desired_perims = np.array([desired_perims])  # np.random.normal(loc=mu_p, scale=std_p, size=N)

    # desired_areas = np.array([123.63451648]) # np.random.normal(loc=mu_a, scale=std_a, size=N)
    desired_areas = np.array([desired_areas])  # np.random.normal(loc=mu_a, scale=std_a, size=N)

    #desired_perims = np.random.shuffle(desired_perims)
    #desired_areas = np.random.shuffle(desired_areas)
    #desired_perims = desired_perims[0]
    #desired_areas = desired_areas[0]
    #desired_areas = np.random.choice(total_area_perim[:,0],size=1)
    #desired_perims = np.random.choice(total_area_perim[:,1],size=1)
    print("Desired Area {}".format(desired_areas))
    print("Desired Perimeter {}".format(desired_perims))


    #tot = np.empty((N, 2))
    #tot[:, 0] = 38
    #tot[:, 1] = 129

    R = areas#perims**2/areas
    alpha = 10

    s = dirichlet.rvs([alpha,alpha,alpha,alpha,alpha], size=1, random_state=1)
    S = desired_areas*s
    S = S[0]
    ss = 0
    for s_i in S:
        ss += 2*np.sqrt(np.pi*s_i)

    d_tp = desired_perims - ss


    p = dirichlet.rvs([alpha,alpha,alpha,alpha,alpha],size=1,random_state=1)
    # Set P_i = 2 \sqrt {\pi *S_i} + p_i * D_TP.
    P = 2*np.sqrt(np.pi*S)+p*d_tp
    P = P[0]
    # 4 Compute R_i = P_i^2 / S_i.
    R_des = S#P**2/S
    R_des = R_des.reshape(5)
    #print(R.shape)
    #print(R_des.shape)
    index = np.empty((5))
    for i in range(5):
        _, ind = find_nearest(R,R_des[i])
        R = np.delete(R,ind)

        index[i] = ind
    index = index.astype(int)
    #index = np.array(index)
    #index = index.flatten()
    #print(index)
    #print(nearest_in_dataset)

    #xy_ind = list(chain.from_iterable((i, i + 1) for i in range(0, 50, 10)))
    #xy_ind.append(50)
    #xy_ind.append(51)
    #xy_ind.append(52)
    #print(xy_ind)
    #all_ind = np.array(range(53))
    #take_ind = np.setdiff1d(all_ind,xy_ind)
    #print(len(take_ind))
    #w_xyrad = w_train_red[:, take_ind]
    w_rad = w_train_red[:,:-3]
    w_rad = w_rad.reshape(w_rad.shape[0]*5,11)
    #print(w_rad.shape)
    #print(w_xyrad.shape[0]*5)
    #print(index)
    #single_platforms = np.empty((R.shape[0],8))
    #for row in range(w_xyrad.shape[0]):
    #    single_platforms[5*row,:] = w_xyrad[row,:8]
    #    single_platforms[5*row+1,:] = w_xyrad[row,8:16]
    #    single_platforms[5*row+2,:] = w_xyrad[row,16:24]
    #    single_platforms[5*row+3,:] = w_xyrad[row,24:32]
    #    single_platforms[5*row+4,:] = w_xyrad[row,32:40]

    #w_xyrad = w_xyrad.flatten()
    #print(w_xyrad.shape)
    #print(R.shape)
    # %%

    #w_xyrad = np.reshape(w_xyrad, (-1, 10))
    #platforms = np.empty((5,w_xyrad.shape[1]+3))
    platforms = np.take(w_rad,index,axis=0)
    #print(platforms.shape)
    #platforms_to_plot = np.empty((1,53))
    #for ind, val in zip(take_ind,platforms.flatten()):
    #    platforms_to_plot[:,ind] = val
    #platforms_to_plot[:,take_ind] = platforms.flatten()

    #mean_xy = np.mean(w_train_red[:,xy_ind], axis=0)
    #cov_xyz = np.cov(w_train_red[:,xy_ind], rowvar=0)
    #platforms_to_plot[:,xy_ind] = np.random.multivariate_normal(mean=mean_xy,cov=cov_xyz)


    #plot_platforms(w_train_red[0,:])
    #plat_plot = np.zeros((1,58))
    #plat_plot[:,:-3] = platforms.reshape(1,55)
    #ut.plot_contours_sdf(w_train_red[0,:],v=5,u=8)

    all_possible_inds = []
    # compute the unique possible positions
    #w_bool = w_train_red.astype(bool)
    ##print(w_bool)
    #w_unique = np.unique(w_bool,axis=0)
    #print("Number of unique configurations {}".format(w_unique.shape[0]))
    #supports = load_grids()
    #all_possible_inds = []
    #for i in range(5):
    #    possible_indices = possible_permutations(platforms[i], w_unique)
    #    all_possible_inds.append(possible_indices)

    return platforms, desired_areas, desired_perims

#if __name__ == "__main__":
#    main()
