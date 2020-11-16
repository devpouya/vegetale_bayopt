#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:43:21 2020

@author: luissalamanca
"""

"""
In this script we present some basic snippets to open the data, convert it, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle as pk
import time
import pandas as pd

import utils as ut
import time

import copy

import scipy

#%%

# constants
# rhino uv nurbs +1 (even numbers required?
u = 8 #u = number of points 
v = 5 #v = number of platforms
#f_experiment = '200616'
curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/' #+ f_experiment
results_dir = curr_dir_repo + '/results/' #+ f_experiment

fn_attr_ranges = 'ranges_attrib.pkl'

###
# THE NOTATION IS
# W for geometries: the points, either on xyz or polar, for each of the platforms
# X for desing attributes: final desired characteristics, like rain occlusion,
# sun occlusion, surface, etc.

flag_polar = True

    
#%%
# OLD VERSION OF DATASET LOADING
"""
# Load data
# Loading the 230k samples generated already with GH
# load datasets
x_dir = os.path.join(dataset_dir, 'labels')

# Geometries
w_dir = os.path.join(dataset_dir, 'xyz')
w_fname = os.path.join(w_dir, 'data_xyz_all.pkl')

w_train = pk.load(open(w_fname,'rb'))

num_samples = len(w_train)

#%%
# Import and preprocess x_train: geometries

w_train = np.asarray(w_train) # cast into array

# Reduce from 120 points (3 xyz, 8 for each of the 5 platforms : 5 * 8 * 3), to
# 83 dimensions, as we only keep one z per platform, except for the first
# of the last, that are fixed (2 * 8 * 5 + (v - 2))
w_train_red, bottom_top_heights = ut.reduce_xyz(w_train, v, u, num_samples)

#%%
# Convert to polars, as they are less dimensions, only 53:
# 8 radius per platform, and 2 xy for the center of each platform. Plus v - 2
# heights
if flag_polar:
    w_train_red_xyz = copy.copy(w_train_red)
    w_train_red = ut.convert_topolars(w_train_red_xyz, v, u, num_samples)
    
"""

#%%
# NEW VERSION
# This data now also includes surface and perimeter per platform
# data_labels_all_200702 includes around 80k samples
# data_labels_all_200706 includes around 200k samples
fname = os.path.join(dataset_dir, 'data_labels_all_200706.pkl')
list_all_dicts = pk.load(open(fname, 'rb'))

#%%

flag_GH_xyz_polar = 0
flag_out_xyz_polar = 0
if flag_polar:
    flag_GH_xyz_polar = 1
    flag_out_xyz_polar = 1
    
w_train_red, bottom_top_heights = ut.gh_to_script(list_all_dicts, v, u, flag_GH_xyz_polar = flag_GH_xyz_polar,
                                             flag_out_xyz_polar = flag_out_xyz_polar)
print(bottom_top_heights)

#%%
# Import and preprocess y_train: design attributes


# Attributes to use
list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3', 'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']
x_train = ut.ghlabels_to_script(list_all_dicts, list_attrib,flag_all_df = True)

all_attributes = list(x_train.columns)
print(all_attributes)
#%%



num_attr = len(list_attrib) # number of attributes selected for training

#print("Attributes",list_attrib)

x_train_red = np.asarray(x_train[list_attrib])

#%%
# Check for possible nans

w_nans, ind_not_nanw = ut.check_rows_nan(w_train_red)
x_nans, ind_not_nanx = ut.check_rows_nan(x_train_red)

if len(np.union1d(w_nans, x_nans)):
    ind_not_nan = np.intersect1d(ind_not_nanw, ind_not_nanx)
    print('Reducing samples from {} to {}'.format(len(w_train_red), len(ind_not_nan)))
    w_train_red = w_train_red[ind_not_nan, :]
    x_train_red = x_train_red[ind_not_nan, :]
else:
    print('No nans! ¯\_(ツ)_/¯')
    
#%%
# Plot some set of geometries
"""
ut.plot_scatter_geometries(w_train_red, v, u, numb_samp = 10)

#%%

ut.plot_contours(w_train_red, v, u, x_vals = x_train_red,
                 list_att = ['OccRain','OccSun','Surf','Outline'], list_plot_x = [[0,1],[2,3]],
                 samp_to_plot = 1000)
"""
#%%
# Plotting design attributes from GH samples
list_attrib_aux = ['occlusion_rain', 'occlusion_sun', 'surface_area_total', 'outline_length_total']
"""
ind_val = [0,1]
x_train_red_full = np.asarray(x_train[list_attrib_aux])
plt.scatter(x_train_red_full[::100,ind_val[0]],x_train_red_full[::100,ind_val[1]])
plt.xlabel(list_attrib_aux[ind_val[0]])
plt.ylabel(list_attrib_aux[ind_val[1]])
plt.xlim([5,90])
plt.ylim([5,90])

plt.figure()
ind_val = [2,3]
x_train_red_full = np.asarray(x_train[list_attrib_aux])
plt.scatter(x_train_red_full[::100,ind_val[0]],x_train_red_full[::100,ind_val[1]])
plt.xlabel(list_attrib_aux[ind_val[0]])
plt.ylabel(list_attrib_aux[ind_val[1]])
plt.xlim([80,190])
plt.ylim([80,190])

plt.figure()
ind_val = [0,2]
x_train_red_full = np.asarray(x_train[list_attrib_aux])
plt.scatter(x_train_red_full[::100,ind_val[0]],x_train_red_full[::100,ind_val[1]])
plt.xlabel(list_attrib_aux[ind_val[0]])
plt.ylabel(list_attrib_aux[ind_val[1]])

plt.figure()
ind_val = [1,3]
x_train_red_full = np.asarray(x_train[list_attrib_aux])
plt.scatter(x_train_red_full[::100,ind_val[0]],x_train_red_full[::100,ind_val[1]])
plt.xlabel(list_attrib_aux[ind_val[0]])
plt.ylabel(list_attrib_aux[ind_val[1]])
"""
#%%
# Generate file for GH, and wait until it generates backs

data_dir_loop = '/Users/pouya/vegetale_bayopt'
str_save = 'It1'
flag_save_files = True
max_time = 3600
print(w_train_red.shape)
print(x_train_red.shape)
list_dicts = ut.polar_to_ghpolar(w_train_red, bottom_top_heights, 
                                 v, u, flag_create_dict = True,
                                 x_in = x_train_red, list_attrib = list_attrib,
                                 flag_save_files = flag_save_files, str_save = str_save)

list_dicts = list_dicts[:10]
print(list_dicts[0]["labels"])
f_name = os.path.join(data_dir_loop,'tmp_fromscript.pkl')
pk.dump(list_dicts, open(f_name,'wb'), protocol=2)

# Since grasshopper removes the file when it read it, we can save it in a different 
# folder, in case we need it for later
if len(str_save):
    f_name = os.path.join(data_dir_loop,'saved/tmp_fromscript_{}.pkl'.format(str_save))
    pk.dump(list_dicts, open(f_name,'wb'), protocol=2)

# Waiting for the generation of the new geometries
t_st = time.time()
f_name_out = os.path.join(data_dir_loop,'tmp_fromgh.pkl')
print('Waiting for file...')
dict_polar = ut.wait_gh_x(f_name_out, max_time)
t_end = time.time() - t_st
print('Finished the loop on GH for %d samples in total time of %.2f'%(len(list_dicts),t_end))
print(dict_polar[0]["labels"])
#print(dict_polar["labels"])
# Convert the dict to the format required. In this case, we just need
# the labels
x_gh = ut.ghlabels_to_script(dict_polar, list_attrib, flag_all_df = True)