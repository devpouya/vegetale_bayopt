import os, sys
#sys.path.append('../src/python/')
sys.path.insert(1, '/Users/pouya/vegetale_bayopt/external/combo/COMBO')
import copy
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import utils_vegetale as uv
import generate_platform as gen

from external.combo.COMBO import main


#file_data = 'data_labels_SDF_200830_470K_cl_flags_occl.pkl'
#dataset_dir = ''

#fname = os.path.join(dataset_dir, file_data)
#list_all_dicts = pk.load(open(fname, 'rb'))

#file_grid = './data/grid/grid_points_coord.pkl'
#grid_pts_xy = pk.load(open(os.path.join(dataset_dir, file_grid), 'rb'))

v = 5
u = 8
# w is a list where each element corresponds to one sample. The vector for each samples
# contains first the 5 * 11 radii. From the bottom to the top platform. Remember that
# radii equal to 0 are inactive poles
# The last 3 elements are the heights of the intermediate platforms, that is constant
#w, bottom_top_heights = uv.gh_to_script(list_all_dicts, v, 8, flag_sdf = True)


# what i need to do
# load the file
# pick perimeter and surface

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


def main():
    file_data = 'data_labels_SDF_200830_470K_cl_flags_occl.pkl'
    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    tot_area_perim_attr = ['surface_area_total', 'outline_length_total']

    w_train_red, x_train, total_area_perim = gen.load_raw_data(list_attrib, tot_area_perim_attr, fname_string=file_data)
    w_train_red = w_train_red[:,:-3]
    w_train_red = w_train_red.reshape(w_train_red.shape[0]*5,11)
    print(w_train_red.shape)
    w_bool = w_train_red.astype(bool)
    w_unique = np.unique(w_bool, axis=0)
    print(w_unique.shape)
    all_possible_inds = []
    #for i in range(w_train_red.shape[0]):
    #    print(i)
    #    possible_indices = gen.possible_permutations(w_train_red[i,:], w_unique)
    #    all_possible_inds.append(possible_indices)

    #with open("search_space.pkl", "wb") as fp:
    #    pk.dump(all_possible_inds, fp)


    rain_des = 25  # float(input("Please enter the desired rain_occlusion: "))
    sun_des = 25  # float(input("Please enter the desired sun_occlusion: "))

    platforms, _ = gen.generate(file_data)
    for i in range(5):
        possible_indices = gen.possible_permutations(platforms[i,:], w_unique)
        all_possible_inds.append(possible_indices)
    #gen.read
    #
    for s in all_possible_inds:
        print(len(s))
    #print(len(all_possible_inds))
    #print(all_possible_inds)
    #print(platforms)
    #print(platforms.shape)
    #supports = load_grids()
    #print(supports)
    #print(supports.shape)





if __name__ == "__main__":
    main()


