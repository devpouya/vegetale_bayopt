import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle as pk
import time
import pandas as pd
import utils as ut
import copy
import scipy
import math
import scipy.stats as stats
import sklearn

u = 8  # u = number of points
v = 5  # v = number of platforms

curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/'  # + f_experiment
results_dir = curr_dir_repo + '/results/'  # + f_experiment

###
# THE NOTATION IS
# W for geometries: the points, either on xyz or polar, for each of the platforms
# X for desing attributes: final desired characteristics, like rain occlusion,
# sun occlusion, surface, etc.

# use polar coordinates for values
flag_polar = True

# load the small dataset (80K samples)
#
fname = os.path.join(dataset_dir, 'data_labels_all_200706.pkl')
list_all_dicts = pk.load(open(fname, 'rb'))

flag_GH_xyz_polar = 0
flag_out_xyz_polar = 0
if flag_polar:
    flag_GH_xyz_polar = 1
    flag_out_xyz_polar = 1

w_train_red, bottom_top_heights = ut.gh_to_script(list_all_dicts, v, u, flag_GH_xyz_polar=flag_GH_xyz_polar,
                                                  flag_out_xyz_polar=flag_out_xyz_polar)

list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3', 'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']
x_train = ut.ghlabels_to_script(list_all_dicts, list_attrib,flag_all_df = True)

all_attributes = list(x_train.columns)
#print(all_attributes)

from itertools import chain

perimeter_surface = x_train[['outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2',
                             'outline_lengths_3', 'outline_lengths_4', 'surface_areas_0', 'surface_areas_1',
                             'surface_areas_2',
                             'surface_areas_3', 'surface_areas_4']]
list_bop = ['outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3', 'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

xy_ind = list(chain.from_iterable((i, i + 1) for i in range(10, 50, 10)))
h_ind = [50, 52, 51]
x_train_red = np.asarray(x_train[list_attrib])
w_nans, ind_not_nanw = ut.check_rows_nan(w_train_red)
x_nans, ind_not_nanx = ut.check_rows_nan(x_train_red)
if len(np.union1d(w_nans, x_nans)):
    ind_not_nan = np.intersect1d(ind_not_nanw, ind_not_nanx)
    #print('Reducing samples from {} to {}'.format(len(w_train_red), len(ind_not_nan)))
    w_train_red = w_train_red[ind_not_nan, :]
    x_train_red = x_train_red[ind_not_nan, :]
#else:
    #print('No nans! ¯\_(ツ)_/¯')

w_train_mod = w_train_red
#w_train_mod[:, xy_ind] = np.random.normal(size=len(xy_ind))
#w_train_mod[:, h_ind] = np.random.normal(size=len(h_ind))


#from bayes_opt import BayesianOptimization
from GPyOpt.methods import BayesianOptimization

# want: given 5 platforms with orientation, surfaces, perimeter predict sun/rain_occlusion
#       input -> 1 row of design to GP
#       output -> sun_occlusion, rain_occlusion
#       GH is blackbox
#       Fixed parameters (14 in total) : 5 platforms (perimeters, surface area)
#                                        PLT0(x,y,h) = (0,0,4.5) (3)
#                                        PLT4_h = 19 (1)
#       Free parameters (11 in total):   (x,y,h) of PLT1-PLT3 (9)
#                                        (x,y) of PLT4 (2)
# Approach 1: Acquisition function tells GH which point in design to evaluate next
# Approach 2: Acquisition function tells GH which free parameters to evaluate next
# do it for each point
# Minimize mean error in (sun_occ, rain_occ)
lower_xy = -3
upper_xy = 3
mu = 0
sigma = 1.0
lower_h = 4.5
upper_h = 19



bounds = {'x1': (lower_xy,upper_xy),
          'y1': (lower_xy, upper_xy),

          'x2':(lower_xy,upper_xy),
          'y2': (lower_xy, upper_xy),

          'x3':(lower_xy,upper_xy),
          'y3': (lower_xy, upper_xy),

          'x4':(lower_xy,upper_xy),
          'y4':(lower_xy,upper_xy),

          'h1':(lower_h,upper_h),
          'h2': (lower_h,upper_h),
          'h3': (lower_h,upper_h)}

bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'y1', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'x2', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'y2', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'x3', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'y3', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'x4', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'y4', 'type': 'continuous', 'domain': (lower_xy,upper_xy)},
          {'name': 'h1', 'type': 'continuous', 'domain': (lower_h,upper_h)},
          {'name': 'h2', 'type': 'continuous', 'domain': (lower_h,upper_h)},
          {'name': 'h3', 'type': 'continuous', 'domain': (lower_h,upper_h)},

          ]

max_iter = 5






from bayes_opt import UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

i = 0

data_dir_loop = '/Users/pouya/vegetale_bayopt'
strn = 'It1'
max_time = 3600
#target_sun = 24
#target_rain = 38
output_attrib = ['occlusion_rain', 'occlusion_sun']
#x_train_red = np.asarray(x_train[list_attrib])

def GH_black_box(params):
    xy_ind = list(chain.from_iterable((i, i+1) for i in range(10, 50,10)))
    h_ind = [50,52,51]
    all_ind = xy_ind+h_ind
    #print(len(kwargs.values()))
    #print(kwargs)
    global i
    global w_train_mod
    global x_train_red
    i = i + 1
    str_save = strn + str(i)
    """
    learned_params = [kwargs["x1"],kwargs["y1"],
                      kwargs["x2"],kwargs["y2"],
                      kwargs["x3"],kwargs["y3"],
                      kwargs["x4"],kwargs["y4"],
                      kwargs["h1"],kwargs["h2"],kwargs["h3"]]
    """
    print("PARAMS {}".format(params))
    print("W_TRAIN_MOD before {}".format(w_train_mod[0,all_ind]))
    w_train_mod[0,all_ind] = params
    print("W_TRAIN_MOD after {}".format(w_train_mod[0,all_ind]))

    #print(w_train_mod[0,:])
    #print(bottom_top_heights)
    #print(perimeter_surface.shape)
    list_dicts = ut.polar_to_ghpolar(w_train_mod, bottom_top_heights,
                                 v, u, flag_create_dict = True,
                                 x_in = x_train_red, list_attrib = list_attrib,
                                 flag_save_files = True, str_save = str_save)
    list_dicts = list_dicts[0]
    f_name = os.path.join(data_dir_loop,'tmp_fromscript.pkl')
    pk.dump(list_dicts, open(f_name,'wb'), protocol=2)
    if len(str_save):
        f_name = os.path.join(data_dir_loop,'saved/tmp_fromscript_{}.pkl'.format(str_save))
        pk.dump(list_dicts, open(f_name,'wb'), protocol=2)
    t_st = time.time()
    f_name_out = os.path.join(data_dir_loop,'tmp_fromgh.pkl')
    print('Waiting for file...')
    dict_polar = ut.wait_gh_x(f_name_out, max_time)
    t_end = time.time() - t_st
    #print('Finished the loop on GH for %d samples in total time of %.2f'%(len(list_dicts),t_end))
    #print("dict_polar {}".format(dict_polar))
    x_gh = ut.ghlabels_to_script(dict_polar, output_attrib, flag_all_df = True)[output_attrib]
    #x_train = sklearn.preprocessing.normalize(x_train_red)
    xgh = np.asarray(x_gh.values)
    #xgh = sklearn.preprocessing.normalize(xgh)

    calc_sun = xgh[0,0]
    calc_rain = xgh[0,1]
    tr = x_train_red[0,0]
    ts = x_train_red[0,1]
    #print("True rain occ {}".format(calc_rain))
    #print("True sun occ {}".format(calc_sun))
    #print("Predicted occ sun {}".format(ts))
    #print("Predicted occ rain {}".format(tr))
    loss = math.sqrt((calc_rain-tr)**2+(calc_sun-ts)**2)/2
    #print("COCAINE {}".format(loss))
    #print(loss.shape)
    return loss

optimizer = BayesianOptimization(f=GH_black_box,
                                domain = bounds,
                                model_type='GP',
                                acquisition_type ='EI',
                                maximize=False)
optimizer.run_optimization(max_iter=2,verbosity=True)
print("The minumum value obtained by the function was {} (x = {})".format(optimizer.fx_opt, optimizer.x_opt))

optimizer.plot_acquisition()

"""
for _ in range(5):
    next_point = optimizer.suggest(utility)
    target = GH_black_box(**next_point)
    print(next_point)
    print(target)
    optimizer.register(params=next_point, target=target)

    print(target, next_point)
print(optimizer.max)
"""