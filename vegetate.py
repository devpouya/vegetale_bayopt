import os
# sys.path.append('../src/python/')
# sys.path.insert(1, '/Users/pouya/vegetale_bayopt/external/combo/COMBO')
import pickle as pk
import numpy as np
import generate_platform as gen
from combo.COMBO import main as combr
from scipy.stats import norm
import src.python.utils_vegetale as uv
import time
import random
import pandas as pd
from sklearn.preprocessing import Normalizer

# file_data = 'data_labels_SDF_200830_470K_cl_flags_occl.pkl'
# dataset_dir = ''

# fname = os.path.join(dataset_dir, file_data)
# list_all_dicts = pk.load(open(fname, 'rb'))

# file_grid = './data/grid/grid_points_coord.pkl'
# grid_pts_xy = pk.load(open(os.path.join(dataset_dir, file_grid), 'rb'))

v = 5
u = 8
# w is a list where each element corresponds to one sample. The vector for each samples
# contains first the 5 * 11 radii. From the bottom to the top platform. Remember that
# radii equal to 0 are inactive poles
# The last 3 elements are the heights of the intermediate platforms, that is constant
# w, bottom_top_heights = uv.gh_to_script(list_all_dicts, v, 8, flag_sdf = True)


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
surf_attrib = ['surface_area_total', 'outline_length_total']

output_attrib = ['occlusion_rain', 'occlusion_sun']


def load_grids(grid_name_string='grid_points_coord.pkl'):
    gridfile = os.path.join(dataset_dir + "/grid/", grid_name_string)
    grids = pk.load(open(gridfile, 'rb'))
    return grids


def GH_attr(x, x_, bottom_top_heights, max_time=3600, data_dir_loop='/Users/pouya/vegetale_bayopt'):
    list_dicts = uv.polar_to_ghpolar(w=x, bottom_top_heights=bottom_top_heights, v=5, u=8, flag_create_dict=True,
                                     x_in=x_, list_attrib=list_attrib, flag_save_files=False,
                                     str_save='', grid_supp=11, flag_sdf=True)
    f_name = os.path.join(data_dir_loop, 'tmp_fromscript.pkl')
    pk.dump(list_dicts, open(f_name, 'wb'), protocol=2)

    t_st = time.time()
    f_name_out = os.path.join(data_dir_loop, 'tmp_fromgh.pkl')
    # print('Waiting for file...')
    dict_polar = uv.wait_gh_x(f_name_out, max_time)
    t_end = time.time() - t_st
    # print('Finished the loop on GH for %d samples in total time of %.2f' % (len(list_dicts), t_end))
    vals = uv.ghlabels_to_script(dict_polar, list_attrib, flag_all_df=True)
    return vals


def _vegetale_score(x, x_, bottom_top_heights, desired_perims, desired_areas):
    vals = GH_attr(x, x_, bottom_top_heights)
    xgh = vals[surf_attrib]
    xgh = np.asarray(xgh.values)
    print("found values {}".format(xgh))
    # xgh = normalizer.transform(xgh)
    # occs = normalizer.transform(np.array([rain_occ, sun_occ]).reshape(1, 2))

    diff_1 = desired_areas - xgh[0, 0]
    diff_2 = desired_perims - xgh[0, 1]

    loss_area = abs(diff_1)
    loss_perim = abs(diff_2)
    # ITER += 1
    # print(loss)
    return loss_area, loss_perim


def main():
    file_data = 'data_labels_SDF_200830_470K_cl_flags_occl.pkl'
    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    tot_area_perim_attr = ['surface_area_total', 'outline_length_total']

    w_train_red, x_train, total_area_perim, bottom_top_heights = gen.load_raw_data(list_attrib,
                                                                                   tot_area_perim_attr,
                                                                                   fname_string=file_data,
                                                                                   flag_sdf=True)

    data_dir_loop = '/Users/pouya/vegetale_bayopt'

    outputs = x_train[output_attrib].to_numpy()
    mur, stdr = norm.fit(outputs[:, 0])
    # print(mur.shape)
    mus, stds = norm.fit(outputs[:, 1])

    rain_des = np.random.normal(mur, stdr, size=1)
    sun_des = np.random.normal(mus, stds, size=1)
    x_ = np.asarray(x_train.values)
    x_ = x_[0, :]
    x_ = x_.reshape(1, x_.shape[0])
    print("RAIN OCC {} ----- SUN OCC {}".format(rain_des, sun_des))

    mu_p, std_p = norm.fit(total_area_perim[:, 1])

    mu_a, std_a = norm.fit(total_area_perim[:, 0])
    N = 1

    # desired_perims = np.random.normal(loc=mu_p, scale=std_p, size=N)

    # desired_area = np.random.normal(loc=mu_a, scale=std_a, size=N)

    desired_area = 260.94
    desired_perims = 123.63

    normalizer = Normalizer().fit(outputs)

    platforms, desired_area, desired_perims = gen.generate_sdf(file_data, w_train_red, x_train,
                                                               total_area_perim,
                                                               desired_perims, desired_area)

    search_space = []
    all_possible_inds = []

    w_train = w_train_red[:, :-3]
    w_train = w_train.reshape(w_train.shape[0] * 5, 11)
    w_bool = w_train.astype(bool)
    w_unique = np.unique(w_bool, axis=0)

    for i in range(5):
        possible_indices, configurations = gen.possible_permutations(platforms[i, :], w_unique)
        all_possible_inds.append(possible_indices)
        search_space.append(configurations)
    print(platforms.shape)

    num_runs = 3
    all_difs = np.zeros((num_runs + 1, 2))
    start = time.time()
    # occ_pairs = [(10,10),(10,35),(30,80),(30,40),(35,70),(20,60),(57.1,24.4)]
    occ_pairs = [(20, 60)]
    # occ_pairs = [(35,70)]
    for sun_des, rain_des in occ_pairs:
        loss_rain = 0
        loss_sun = 0
        for i in range(num_runs):
            seed = int(random.randint(2,25))
            print("IIII {} has rand seed {}".format(i,seed))
            winner_positions = combr.main(objective="vegetale", space=search_space, platforms=platforms,
                                          bottom_top_heights=bottom_top_heights, rain_occ=rain_des, sun_occ=sun_des,
                                          random_seed_config=seed, normalizer=normalizer, x=x_)

            p1 = search_space[0][winner_positions[0]]
            p2 = search_space[1][winner_positions[1]]
            p3 = search_space[2][winner_positions[2]]
            p4 = search_space[3][winner_positions[3]]
            p5 = search_space[4][winner_positions[4]]

            final = np.zeros((1, 58))
            final[:, :11] = p1
            final[:, 11:22] = p2
            final[:, 22:33] = p3
            final[:, 33:44] = p4
            final[:, 44:55] = p5
            # platforms[:, :-3] = platforms_eval.reshape(1, 55)
            final[:, -1] = 15
            final[:, -2] = 11.5
            final[:, -3] = 7.5
            # print(len(all_possible_inds))
            # print(all_possible_inds)
            # print(platforms)
            # print(platforms.shape)
            # supports = load_grids()
            # print(supports)
            # print(supports.shape)
            final_name = os.path.join(data_dir_loop,
                                      'final_platform_var_{}_rain_{}_sun_{}.pkl'.format(i, rain_des, sun_des))
            pk.dump(final, open(final_name, 'wb'), protocol=2)
            data_dir_loop = '/Users/pouya/vegetale_bayopt'
            # print(final.shape)
            list_dicts = uv.polar_to_ghpolar(w=final.reshape(1, 58), bottom_top_heights=bottom_top_heights, v=5, u=8,
                                             flag_create_dict=True,
                                             x_in=x_, list_attrib=list_attrib, flag_save_files=False,
                                             str_save='', grid_supp=11, flag_sdf=True)
            f_name = os.path.join(data_dir_loop, 'tmp_fromscript.pkl')
            pk.dump(list_dicts, open(f_name, 'wb'), protocol=2)

            f_name_out = os.path.join(data_dir_loop, 'tmp_fromgh.pkl')
            # print('Waiting for file...')
            dict_polar = uv.wait_gh_x(f_name_out, 3600)
            # print('Finished the loop on GH for %d samples in total time of %.2f' % (len(list_dicts), t_end))
            vals = uv.ghlabels_to_script(dict_polar, list_attrib, flag_all_df=True)

            xgh = vals[output_attrib]
            xgh = np.asarray(xgh.values)
            print("found final values {}".format(xgh))

            print("Desired values were: ")
            print("RAIN OCC {} ----- SUN OCC {}".format(rain_des, sun_des))

            diff_1 = rain_des - xgh[0, 0]
            diff_2 = sun_des - xgh[0, 1]
            all_difs[i, 0] = diff_1
            all_difs[i, 1] = diff_2
            loss_1 = abs(diff_1)
            loss_2 = abs(diff_2)

            print("Divergence {} and {}".format(loss_1, loss_2))
            loss_rain += loss_1
            loss_sun += loss_2
        end = time.time()
        all_time = end - start

        print("TIME IT TOOK FOR {} RUNS {}".format(num_runs, all_time))
        loss_rain /= num_runs
        loss_sun /= num_runs
        print("AVERAGE ERROR RAIN {} and  AVERAGE ERROR SUN {}".format(loss_rain, loss_sun))
        all_difs[-1, 0] = loss_rain
        all_difs[-1, 1] = loss_sun
        final_name = os.path.join(data_dir_loop, 'var_final_diffs_rain_{}_sun_{}.pkl'.format(rain_des, sun_des))
        pk.dump(all_difs, open(final_name, 'wb'), protocol=2)


if __name__ == "__main__":
    main()
