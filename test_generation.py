import numpy as np
import generate_platform as gen
import src.python.utils_vegetale as uv
import os
import pickle as pk
import time
from scipy.stats import norm, dirichlet, multivariate_normal
import itertools

curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/'  # + f_experiment
results_dir = curr_dir_repo + '/results/'  # + f_experiment
data_dir_loop = '/Users/pouya/vegetale_bayopt'

file_data_xyz = 'data_labels_all_200706.pkl'
file_data_sdf = 'data_labels_SDF_200830_470K_cl_flags_occl.pkl'
grid_order = 'grid_points_coord.pkl'

list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']
surf_attrib = ['surface_area_total', 'outline_length_total']

output_attrib = ['occlusion_rain', 'occlusion_sun']

def GH_attr(x, x_,bottom_top_heights, max_time=3600, data_dir_loop='/Users/pouya/vegetale_bayopt'):
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


def _vegetale_score(x, x_,bottom_top_heights,desired_perims,desired_areas):
    vals = GH_attr(x, x_,bottom_top_heights)
    xgh = vals[surf_attrib]
    xgh = np.asarray(xgh.values)
    print("found values {}".format(xgh))
    #xgh = normalizer.transform(xgh)
    #occs = normalizer.transform(np.array([rain_occ, sun_occ]).reshape(1, 2))

    diff_1 = desired_areas - xgh[0, 0]
    diff_2 = desired_perims - xgh[0, 1]

    loss_area = abs(diff_1)
    loss_perim = abs(diff_2)
    # ITER += 1
    #print(loss)
    return loss_area, loss_perim

# plot example

desired_area = 260.94
desired_perims = 123.63
#w_train_red, x_train, total_area_perim,_ = gen.load_raw_data(list_attrib, surf_attrib,
#                                                         fname_string=file_data_sdf, flag_sdf=True)
#
#mu_p, std_p = norm.fit(total_area_perim[:, 1])
##mean_p = np.mean(perims, axis=0)
##cov_p = np.cov(perims, rowvar=0)
#
#mu_a, std_a = norm.fit(total_area_perim[:, 0])
##mean_a = np.mean(areas, axis=0)
##cov_a = np.cov(areas, rowvar=0)
#N = 1
#
##desired_perims = np.random.normal(loc=mu_p, scale=std_p, size=N)
#
##desired_area = np.random.normal(loc=mu_a, scale=std_a, size=N)
#
#platforms_sdf, _, _ = gen.generate_sdf(file_data_sdf, w_train_red, x_train,
#                                       total_area_perim, desired_perims, desired_area)
#print(platforms_sdf)
#pz = np.zeros((1, 58))
#pz[0, :55] = platforms_sdf.flatten()
#pz[0,55] = 6.5
#pz[0,56] = 8.5
#pz[0,57] = 10.5
#print(pz)
#fname = os.path.join(dataset_dir, grid_order)
##order_grid = pk.load(open(fname, 'rb'))
#order_grid = pk.load(open(fname, 'rb'))
#
#np.array([4.5,19])
##la, lp = _vegetale_score(pz, x_train, np.array([4.5,19]), desired_perims, desired_area)
#val = GH_attr(pz,x_train,np.array([4.5,19]))

#uv.plot_contours_sdf(pz,5,11,order_grid)




bottom_top_heights = np.array([4.5,19])
#val_pairs = [(250, 140), (325, 170), (150, 110), (325, 110), (150, 170), (300, 160)]
val_pairs = [(260.94, 123.63)]

results = np.zeros((6,2))

w_train_red, x_train, total_area_perim,_ = gen.load_raw_data(list_attrib, surf_attrib,
                                                         fname_string=file_data_xyz, flag_sdf=False)
for iter, (desired_area, desired_perims) in enumerate(val_pairs):
    loss_area = 0
    loss_perim = 0

    for i in range(1):
       platforms_sdf,_, _ = gen.generate_sdf(file_data_sdf, w_train_red, x_train,
                                                       total_area_perim,desired_perims, desired_area)
       #platforms_xyz, _ ,_, _ = gen.generate_xyz(file_data_xyz, desired_area, desired_perims)
       pz = np.zeros((1,58))
       pz[0,:55] = platforms_sdf.flatten()
       la,lp = _vegetale_score(pz,x_train,bottom_top_heights,desired_perims,desired_area)
       loss_area += la
       loss_perim += lp

    loss_area /= 1
    loss_perim /= 1
    results[iter,0] = loss_area
    results[iter,1] = loss_perim
    print("LOSS OF 20 runs area: {} perim: {}".format(la,lp))

print(results)
final_name = os.path.join(data_dir_loop, 'final_res_sdf_gen_full.pkl')
pk.dump(results, open(final_name, 'wb'), protocol=2)
