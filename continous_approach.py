import os
#sys.path.append('../src/python/')
#sys.path.insert(1, '/Users/pouya/vegetale_bayopt/external/combo/COMBO')
import pickle as pk
import numpy as np
import generate_platform as gen
from combo.COMBO import main as combr
from scipy.stats import norm
import src.python.utils_vegetale as uv
#import src.python.move_platforms_bo as mvp
import time
from GPyOpt.methods import BayesianOptimization
from src.python import utils as ut
from sklearn.preprocessing import normalize
from itertools import chain


list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

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
    x_rain_sun = x_ground_truth[:, [rain_ind,sun_ind]]
    x_rain_sun_normalized = normalize(x_rain_sun)

    return x_rain_sun, x_rain_sun_normalized

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
    res = np.empty((1, 1))
    res[0, 0] = adl + pos_const
    return res

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
    print(X_step)
    print(Y_step)
    print("\n\nProcessing sample {}\n\n".format(index))
    min_loss = loss
    context = {}
    max_iter = 100
    while tol < loss and current_iter < max_iter:
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
        fixed_platforms = X_step[np.argmin(Y_step), :]
        if min_loss < 0.01:
            improvement = 0.019

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

def check_difference_results(w_train_red,bottom_top_heights,all_inds,x_ground_truth,x_rain_sun,output_attrib,results,index=0):
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
    diff_1 = abs(true_occ1 - xgh[0, 0])
    diff_2 = abs(true_occ2 - xgh[0, 1])
    print("Difference OCC 1 {}".format(diff_1))
    print("Difference OCC 2 {}".format(diff_2))

    return diff_1, diff_2

def main():
    file_data = 'data_labels_all_200706.pkl'

    list_attrib = ['occlusion_rain', 'occlusion_sun',
                   'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
                   'outline_lengths_4',
                   'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

    tot_area_perim_attr = ['surface_area_total', 'outline_length_total']

    output_attrib = ['occlusion_rain', 'occlusion_sun']

    #w_train_red, x_train, total_area_perim, bottom_top_heights = gen.load_raw_data(list_attrib,
    #                                                                               tot_area_perim_attr,
    #                                                                               fname_string=file_data)
    #print(bottom_top_heights)
    #w_train_red = w_train_red[:,:-3]
    #w_train_red = w_train_red.reshape(w_train_red.shape[0]*5,11)

    #w_bool = w_train_red.astype(bool)

    #w_unique = np.unique(w_bool, axis=0)

    #all_possible_inds = []

    data_dir_loop = '/Users/pouya/vegetale_bayopt'



    #outputs = x_train[output_attrib].to_numpy()
    #mur, stdr = norm.fit(outputs[:,0])

    #mus, stds = norm.fit(outputs[:,1])

    #rain_des = np.random.normal(mur,stdr,size=1)
    #sun_des = np.random.normal(mus,stds,size=1)
    #x_ = np.asarray(x_train.values)
    #x_ = x_[0,:]
    #x_ = x_.reshape(1,x_.shape[0])
    #print("RAIN OCC {} ----- SUN OCC {}".format(rain_des,sun_des))


    #normalizer = Normalizer().fit(outputs)
    platforms, w_train_red ,x_train,desired_area,desired_perims= gen.generate_xyz(file_data)
    fixed_platforms = np.zeros((1,53))
    fixed_platforms[0,:40] = platforms.flatten()

    w_stacked = stack_platforms(fixed_platforms)


    xy_ind = list(chain.from_iterable((i, i + 1) for i in range(10, 50, 10)))
    h_ind = [50, 52, 51]
    all_inds = xy_ind + h_ind
    lower_xy = -6
    upper_xy = 6

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
    bottom_top_heights = np.array([4.5,19])

    occ_pairs = [(10,10),(10,35),(30,80),(30,40),(35,70),(20,60),(57.1,24.4)]
    num_tries = 5
    index = 0
    x_train = np.asarray(x_train)
    all_difs = np.zeros((num_tries+1,2))
    for sun_des, rain_des in occ_pairs:
        loss_rain = 0
        loss_sun = 0
        x_train[0,0] = rain_des
        x_train[0,1] = sun_des
        x_rain_sun, x_rain_sun_normalized = get_xrainsun(x_train, rain_ind=0, sun_ind=1)
        for i in range(num_tries):
            results, gp_results = run_bo_(w_stacked, x_train, x_rain_sun_normalized, bottom_top_heights,
                                          output_attrib, all_inds, bounds, tol=0.009, samples_to_optimize=index)
            df1, df2 = check_difference_results(w_stacked,bottom_top_heights,all_inds
                                                ,x_train,x_rain_sun,output_attrib,results,index=index)
            all_difs[i, 0] = df1
            all_difs[i, 1] = df2
            loss_rain+= df1
            loss_sun+=df2

        loss_rain/=num_tries
        loss_sun/=num_tries

        all_difs[-1, 0] = loss_rain
        all_difs[-1, 1] = loss_sun
        final_name = os.path.join(data_dir_loop, 'CONT_final_diffs_rain_{}_sun_{}.pkl'.format(rain_des, sun_des))
        pk.dump(all_difs, open(final_name, 'wb'), protocol=2)







if __name__ == "__main__":
    main()