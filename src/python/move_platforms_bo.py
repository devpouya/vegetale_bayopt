import numpy as np
import os
import pickle as pk



import utils as ut
import time

from sklearn.preprocessing import normalize
from itertools import chain
from GPyOpt.methods import BayesianOptimization


curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/'  # + f_experiment
results_dir = curr_dir_repo + '/results/'  # + f_experiment
list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

def load_raw_data(list_attrib,u = 8, v = 5, flag_polar=True, fname_string='data_labels_all_200706.pkl'):

    print("Loading raw data...")
    ###
    # THE NOTATION IS
    # W for geometries: the points, either on xyz or polar, for each of the platforms
    # X for desing attributes: final desired characteristics, like rain occlusion,
    # sun occlusion, surface, etc.

    fname = os.path.join(dataset_dir, fname_string)
    list_all_dicts = pk.load(open(fname, 'rb'))


    flag_GH_xyz_polar = 0
    flag_out_xyz_polar = 0
    if flag_polar:
        flag_GH_xyz_polar = 1
        flag_out_xyz_polar = 1

    w_train_red, bottom_top_heights = ut.gh_to_script(list_all_dicts, v, u, flag_GH_xyz_polar=flag_GH_xyz_polar,
                                                      flag_out_xyz_polar=flag_out_xyz_polar)



    x_train = ut.ghlabels_to_script(list_all_dicts, list_attrib, flag_all_df=True)

    all_attributes = list(x_train.columns)
    print(all_attributes)



    x_train = x_train[list_attrib]
    x_train_red = np.asarray(x_train)

    w_nans, ind_not_nanw = ut.check_rows_nan(w_train_red)
    x_nans, ind_not_nanx = ut.check_rows_nan(x_train_red)

    if len(np.union1d(w_nans, x_nans)):
        ind_not_nan = np.intersect1d(ind_not_nanw, ind_not_nanx)
        print('Reducing samples from {} to {}'.format(len(w_train_red), len(ind_not_nan)))
        w_train_red = w_train_red[ind_not_nan, :]
        x_train_red = x_train_red[ind_not_nan, :]
    else:
        print('No nans! ¯\_(ツ)_/¯')

    return w_train_red, x_train_red, bottom_top_heights

def is_within_range(params,ranges):
    res = True
    for tmp in enumerate(zip(params,ranges)):
        p = tmp[0]
        r = tmp[1]
        l,h = r["domain"]
        if p >= l and p <= h:
            continue
        else:
            res = False
            break
    return res

def stack_platforms(w_train_red):
    print("Stacking platforms...")
    xy_ind = list(chain.from_iterable((i, i + 1) for i in range(10, 50, 10)))

    h50 = np.mean(w_train_red[:, 50])
    h51 = np.mean(w_train_red[:, 51])
    h52 = np.mean(w_train_red[:, 52])
    w_train_red[:, xy_ind] = np.random.normal(size=(w_train_red.shape[0], len(xy_ind)))
    w_train_red[:, 50] = h50
    w_train_red[:, 51] = h51
    w_train_red[:, 52] = h52

    return w_train_red

def get_xrainsun(x_train_red, rain_ind, sun_ind):
    x_ground_truth = x_train_red
    x_rain_sun = x_ground_truth[:, [rain_ind, sun_ind]]
    x_rain_sun_normalized = normalize(x_rain_sun)

    return x_rain_sun, x_rain_sun_normalized


def abs_diff_loss(xgh, x_rain_sun , index):
    x_norm = normalize(xgh)

    rain_err = abs(x_norm[0, 0] - x_rain_sun[index, 0])
    sun_err = abs(x_norm[0, 1] - x_rain_sun[index, 1])
    adl = (rain_err + sun_err) / 2
    pos_const = 0

    # add penalty if the obtained values surpass the true
    if (x_rain_sun[index, 0] - x_norm[0, 0]) < 0:
        pos_const += x_norm[0, 0] - x_rain_sun[index, 0]

    if (x_rain_sun[index, 1] - x_norm[0, 1]) < 0:
        pos_const += x_norm[0, 1] - x_rain_sun[index, 1]

    #print("ERROR {} in Garden {}".format(adl, index))
    # true_occ1-xgh[0,0] this has to be positive, penalize (xgh[0,0] - true_occ1) if < 0
    res = np.empty((1, 1))
    res[0, 0] = adl + pos_const
    return res


def GH_attr(w_, x_, bottom_top_heights,max_time=3600,data_dir_loop = '/Users/pouya/vegetale_bayopt'):
    list_dicts = ut.polar_to_ghpolar(w_, bottom_top_heights,
                                     v=5, u=8, flag_create_dict=True,
                                     x_in=x_, list_attrib=list_attrib,
                                     flag_save_files=True, str_save="str_save")

    f_name = os.path.join(data_dir_loop, 'tmp_fromscript.pkl')
    pk.dump(list_dicts, open(f_name, 'wb'), protocol=2)

    t_st = time.time()
    f_name_out = os.path.join(data_dir_loop, 'tmp_fromgh.pkl')
    # print('Waiting for file...')
    dict_polar = ut.wait_gh_x(f_name_out, max_time)
    t_end = time.time() - t_st
    # print('Finished the loop on GH for %d samples in total time of %.2f' % (len(list_dicts), t_end))
    x = ut.ghlabels_to_script(dict_polar, list_attrib, flag_all_df=True)
    return x


def black_box_function(index,all_ind,x_,x_rain_sun,w_train_red,bottom_top_heights,output_attrib, params):

    w_ = w_train_red[index, :]
    w_ = w_.reshape(1, w_.shape[0])
    w_[0, all_ind] = params.reshape(11, )
    x_in = x_[index, :]
    x_in = x_in.reshape(1, x_in.shape[0])
    x = GH_attr(w_, x_in, bottom_top_heights)

    xgh = x[output_attrib]
    xgh = np.asarray(xgh.values)

    loss = abs_diff_loss(xgh, x_rain_sun,index)

    return loss, xgh, w_train_red

def run_bo_(w_train_red, x_train_red,x_rain_sun, bottom_top_heights,output_attrib,all_inds,bounds,model="GP_MCMC",acquisition="LCB",maximize=False,tol=0.009,samples_to_optimize=1):
    w_ = w_train_red
    x_ = x_train_red
    x_1 = x_train_red[samples_to_optimize, :]
    x_1 = x_1.reshape(1, x_1.shape[0])
    true_occ1 = x_1[0, 0]
    true_occ2 = x_1[0, 1]
    print("TRUE OCCLUSION 1 {}".format(true_occ1))
    print("TRUE OCCLUSION 2 {}".format(true_occ2))

    results = np.empty((1, 11))
    gp_results = np.empty((1, 11))
    index = samples_to_optimize


    X_init = np.random.normal(size=(1, 11))
    Y_init, x_gh, w_ = black_box_function(0, all_inds, x_,x_rain_sun, w_,bottom_top_heights,output_attrib, X_init)
    loss = 1
    current_iter = 0
    X_step = X_init
    Y_step = Y_init
    print("\n\nProcessing sample {}\n\n".format(index))
    min_loss = loss
    context = {}
    ind_to_fix = 1
    ind_x_fix = 0
    ind_y_fix = 1
    no_more_fix = False
    fix_h = False
    improvement = 0.5
    while tol < loss:
        #if current_iter > 2:
        #    improvement = 0.05
        print("TOL {} and LOSS {}".format(tol,loss))
        print("ITER {} for sample {}".format(current_iter, index))

        bo_step = BayesianOptimization(f=None, model=model, domain=bounds, X=X_step, Y=Y_step,
                                           maximize=maximize, acquisition=acquisition)
        if len(context) >= 1:
            try:
                x_next = bo_step.suggest_next_locations(context=context)
            except:
                x_next = bo_step.suggest_next_locations()
        else:
            x_next = bo_step.suggest_next_locations()
        y_next, x_gh, w_ = black_box_function(index, all_inds, x_,x_rain_sun, w_,bottom_top_heights,output_attrib,x_next)
        #print("NUM_FIXED = {}".format(len(context)))
        #print("CONTEXT {}".format(context))
        fixed_platforms = X_step[np.argmin(Y_step), :]
        if min_loss < 0.01:
            improvement = 0.019
        #if y_next[0,0] < min_loss:#min_loss - y_next[0,0] > improvement and not no_more_fix:
        #    # fix the next parameter
        #    if not fix_h:
        #        context["x{}".format(ind_to_fix)] = fixed_platforms[ind_x_fix]
        #        context["y{}".format(ind_to_fix)] = fixed_platforms[ind_y_fix]
        #        ind_x_fix += 2
        #        ind_y_fix += 2
        #    else:
        #        context["h{}".format(ind_to_fix)] = fixed_platforms[ind_y_fix]
        #        #ind_x_fix += 2
        #        ind_y_fix += 1

        #    ind_to_fix += 1
        #    if ind_to_fix >= 4:
        #        fix_h = True
        #        ind_to_fix = 1

        x_[index, :2] = x_gh[0, :]
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))

        loss = y_next[0, 0]
        min_loss = min(loss,min_loss)

        current_iter += 1
    print("SMALLEST LOSS {} ".format(np.min(Y_step)))
    #print("OPT PARAMS {}".format(X_step[np.argmin(Y_step), :]))
    results[0, :] = X_step[np.argmin(Y_step), :]

    opt_ind = bo_step.model.predict(bo_step.X)[0].argmin()
    gp_results[0, :] = bo_step.X[opt_ind, :]
    print(bo_step.X[opt_ind, :])
    return results, gp_results



def plot_platforms(w_train_red,x_ground_truth,results,all_inds,u=8,v=5,index=1):
    # i = test
    w_ = w_train_red[index, :].reshape(1, 53)

    ut.plot_contours(w_, v, u, x_vals=x_ground_truth,
                         list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                         samp_to_plot=0)

    pars = results[0]  # results[i,:]
    # i = test
    w_ = w_train_red[index, :].reshape(1, 53)
    w_[0, all_inds] = pars
    ut.plot_contours(w_, v, u, x_vals=x_ground_truth,
                             list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                             samp_to_plot=0)



def check_difference_results(w_train_red,bottom_top_heights,all_inds,x_ground_truth,x_rain_sun,output_attrib,results,index=1):
    pars = results[0]  # results[i,:]
    # i = test
    w_ = w_train_red[index, :].reshape(1, 53)
    w_[0, all_inds] = pars
    x_ = x_ground_truth[index, :]
    x_ = x_.reshape(1, x_.shape[0])
    true_occ1 = x_rain_sun[index, 0]
    true_occ2 = x_rain_sun[index, 1]
    print(true_occ1)
    print(true_occ2)
    # x_[0,0] = np.random.rand(1)
    # x_[0,1] = np.random.rand(1)
    x_mod = GH_attr(w_, np.zeros_like(x_),bottom_top_heights)
    xgh = x_mod[output_attrib]
    xgh = np.asarray(xgh.values)
    diff_1 = true_occ1 - xgh[0, 0]
    diff_2 = true_occ2 - xgh[0, 1]
    print("Difference OCC 1 {}".format(diff_1))
    print("Difference OCC 2 {}".format(diff_2))

    return diff_1, diff_2


def check_difference_gp(w_train_red,bottom_top_heights,all_inds,x_ground_truth,x_rain_sun,output_attrib,gp_results,index=1):
    pars = gp_results[0]  # results[i,:]
    # i = test
    w_ = w_train_red[index, :].reshape(1, 53)
    w_[0, all_inds] = pars
    x_ = x_ground_truth[index, :]
    x_ = x_.reshape(1, x_.shape[0])
    true_occ1 = x_rain_sun[index, 0]
    true_occ2 = x_rain_sun[index, 1]
    # x_[0,0] = np.random.rand(1)
    # x_[0,1] = np.random.rand(1)
    x_mod = GH_attr(w_, x_,bottom_top_heights)
    xgh = x_mod[output_attrib]
    xgh = np.asarray(xgh.values)
    diff_1 = true_occ1 - xgh[0, 0]
    diff_2 = true_occ2 - xgh[0, 1]
    print("Difference OCC 1 {}".format(diff_1))
    print("Difference OCC 2 {}".format(diff_2))

    return diff_1, diff_2


def main():
    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    output_attrib = ['occlusion_rain', 'occlusion_sun']
    index = 17



    w_train_red, x_train_red, bottom_top_heights = load_raw_data(list_attrib)


    w_stacked = stack_platforms(w_train_red)
    x_rain_sun, x_rain_sun_normalized = get_xrainsun(x_train_red,rain_ind=0,sun_ind=1)
    print("X_RAIN_SUN {}".format(x_rain_sun[index,:]))
    xy_ind = list(chain.from_iterable((i, i + 1) for i in range(10, 50, 10)))
    h_ind = [50, 52, 51]
    all_inds = xy_ind + h_ind
    lower_xy = -3
    upper_xy = 3

    lower_h = 4.5
    upper_h = 19

    bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'y1', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'x2', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'y2', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'x3', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'y3', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'x4', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'y4', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
              {'name': 'h1', 'type': 'continuous', 'domain': (lower_h, upper_h)},
              {'name': 'h2', 'type': 'continuous', 'domain': (lower_h, upper_h)},
              {'name': 'h3', 'type': 'continuous', 'domain': (lower_h, upper_h)}
              ]
    occ_pairs = [(10, 35), (30, 80), (30, 40), (35, 70), (10, 80), (20, 60), (57.1, 24.4)]
    num_try = 10
    lossrain = 0
    losssun = 0
    for sun_des, rain_des in occ_pairs:
        all_diffs = np.zeros((num_try,2))
        for i in range(num_try):

            results, gp_results = run_bo_(w_stacked,x_train_red,x_rain_sun_normalized,bottom_top_heights,
                                  output_attrib, all_inds,bounds,tol=0.009,samples_to_optimize=index)

            rd1, rd2 = check_difference_results(w_train_red,bottom_top_heights,all_inds,
                                        x_train_red,x_rain_sun,output_attrib,results,index=index)
            #gd1, gd2 = check_difference_gp(w_train_red,bottom_top_heights,all_inds,
            #                           x_train_red,x_rain_sun,output_attrib,gp_results,index=index)
            lossrain+=rd1
            losssun += rd2
            rd = abs(abs(rd1) - abs(rd2))
            w_stacked[index, all_inds] = results.reshape(11, )
            data_dir_loop = '/Users/pouya/vegetale_bayopt'
            final_name = os.path.join(data_dir_loop, 'CONT_final_platform_{}_rain_{}_sun_{}.pkl'.format(i,rain_des,sun_des))
            pk.dump(w_stacked[index, :], open(final_name, 'wb'), protocol=2)
        lossrain



        #gd = abs(abs(gd1) - abs(gd2))
        #if rd < gd:
        #opt_params = results
        #else:
        #    if is_within_range(gp_results,bounds):
        #        opt_params = gp_results
        #    else:
        #        opt_params = results


    #print("OPTIMAL PARAMETERS {}".format(opt_params))
    #plot_platforms(w_train_red,x_train_red,opt_params,all_inds,index=index)



if __name__ == "__main__":
    main()



