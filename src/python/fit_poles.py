import numpy as np
import os
import pickle as pk

import utils as ut
import move_platforms_bo as mpb
import matplotlib.pyplot as plt
import random

from GPyOpt.methods import BayesianOptimization

from shapely.geometry import Point, Polygon

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


def calc_distance_to_platform(w_geom, v, u, garden_index, w_means=None, w_std=None,
                              grids=None):
    if np.ndim(w_geom) == 1:
        w_geom = w_geom.reshape((1, -1))
    if w_means is not None:
        w_geom = w_geom * w_std + w_means
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_geom, v, u, w_geom.shape[0])

    if grids.ndim < 2:
        grids = grids.reshape(11, 2)

    distances_matrix = np.zeros((grids.shape[0], v))
    inside_dist = []
    outside_dist = []

    for isamp in [garden_index]:
        for iw in range(v):
            ind_use = np.arange(u * 2) + iw * u * 2
            points = w_geom[isamp, ind_use].reshape(u, -1)
            points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
            poly = Polygon(points)

            for pole_index in range(grids.shape[0]):
                pole = Point(grids[pole_index, :])
                if pole.within(poly):
                    dist = poly.exterior.distance(pole)
                    distances_matrix[pole_index, iw] = dist
                    inside_dist.append(dist)
                else:
                    dist = poly.boundary.distance(pole)
                    distances_matrix[pole_index, iw] = dist
                    outside_dist.append(dist)

    return distances_matrix, np.array(inside_dist), np.array(outside_dist)


def count_inside(w_geom, v, u, platform_index, garden_index, w_means=None, w_std=None,
                 grids=None):
    if np.ndim(w_geom) == 1:
        w_geom = w_geom.reshape((1, -1))
    if w_means is not None:
        w_geom = w_geom * w_std + w_means
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_geom, v, u, w_geom.shape[0])
    if grids.size <= 2:
        grids = grids.reshape(1, 2)
    else:
        grids = grids.reshape(int(grids.size / 2), 2)
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)

    count = 0
    inside_index = []
    for isamp in [garden_index]:
        ind_use = np.arange(u * 2) + platform_index * u * 2
        points = w_geom[isamp, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        for i in range(grids.shape[0]):
            pole = Point(grids[i, :])
            if pole.within(poly):
                count += 1
                inside_index.append(i)
            # else:
            #    dist = poly.boundary.distance(pole)
            #    if dist <= 1.5 and dist >= 0.8:
            #        count += 1

    return count, inside_index


def pole_to_pole_distances(w_geom, v, u, w_means=None, w_std=None,
                           samp_to_plot=None, grids=None):
    if np.ndim(w_geom) == 1:
        w_geom = w_geom.reshape((1, -1))
    if w_means is not None:
        w_geom = w_geom * w_std + w_means
    if w_geom.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_geom, v, u, w_geom.shape[0])

    if grids.ndim < 2:
        grids = grids.reshape(grids.shape[0] / 11, 2)
    if samp_to_plot is None:
        vec_rand_samples = np.random.permutation(w_geom.shape[0])
    else:
        vec_rand_samples = [samp_to_plot]
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)
    count = 0
    inside_index = []
    pole_to_pole = np.zeros((grids.shape[0], grids.shape[0]))

    for isamp in vec_rand_samples[:1]:
        for iw in range(v):
            ind_use = np.arange(u * 2) + iw * u * 2
            points = w_geom[isamp, ind_use].reshape(u, -1)
            points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
            poly = Polygon(points)

            for i in range(grids.shape[0]):
                # pole1 = Point(grids[i,:])
                for j in range(grids.shape[0]):
                    # pole2 = Point(grids[j,:])
                    # if pole1.within(poly) and pole2.within(poly):
                    # dist = pole1.distance(pole2)
                    dist = np.linalg.norm(grids[i, :] - grids[j, :])
                    pole_to_pole[i, j] = dist

    return pole_to_pole


def all_poles_dist(grids):
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)
    b = grids.reshape(grids.shape[0], 1, grids.shape[1])

    return np.sqrt(np.einsum('ijk, ijk->ij', grids - b, grids - b))


def count_constraint(grids, w_, garden_index, num_platforms=5, context=None):
    # each platform should have between 3 and 5 poles
    # define constraint s.t it is satisfied if f(x) <= 0
    # a list of constraints, each constraint is a number
    # if all numbers are <= 0, then all constraints are fulfilled
    constraint_list = []
    if context is not None:
        grids = np.array(list(context.values()))
        # print(grids.shape)
        # grids = grids.reshape(grids.shape[0] / 2, 2)
    max_f = -100000
    for i in range(num_platforms):
        count, inside_index = count_inside(w_, v=5, u=8, platform_index=i, garden_index=garden_index, grids=grids)
        penalty = 0

        # f(x) <= 0 then constraint satisfied
        # 3 <= f(x) = count <= 5 constraint satisfied
        # f(x) - 5 <= 0
        # 3 - f(x) <= 0
        smaller_than = 3 - count
        bigger_than = count - 5
        constraint_list.append(smaller_than)
        constraint_list.append(bigger_than)

        max_f = max(max(smaller_than, bigger_than), max_f)
    # constraint_list will be of size 10, each two elements correspond to one platform
    return max_f, np.array(constraint_list)


def all_poles_dist_constraint(grids):
    # the distance between two poles has to be 2 meters
    # f(x) <= 0 --> dist >= 2
    # const: 2 - f(x)
    distances = all_poles_dist(grids)
    m, n = distances.shape
    constraints = 2 - distances
    ret = 19
    if distances.size > 1:
        out = np.arange(m)[:, None] != np.arange(n)
        ret = np.amax(constraints[out])
    else:
        ret = constraints[0]

    return ret, constraints


def poles_platform_distance_constraint(grids, w_, garden_index):
    # if inside dist >= 0.8
    # if outside dist >= 0.8
    distances, inside_dist, outside_dist = calc_distance_to_platform(w_, v=5, u=8, garden_index=garden_index,
                                                                     grids=grids)
    #inside_constraint = 0.8-inside_dist
    #outside_constraint = 0.8-outside_dist
    #inside_max = np.amax(inside_constraint)
    #outside_max = np.amax(outside_constraint)
    constraints = 0.8 - distances
    c_max = np.amax(constraints)
    return c_max, constraints


def plot_platform_grids(w_train_red, x_vals, v=5, u=8,
                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                        samp_to_plot=0, grids=None, pole_ind=None):
    w_ = w_train_red[0, :].reshape(1, 53)
    ut.plot_contours(w_, v=5, u=8, x_vals=x_vals,
                     list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                     samp_to_plot=0, grids=grids, pole_ind=pole_ind)


"""
def all_constraints(w_,garden_index,grids):
    # if return value <= 0 then all constraints are satisfied
    #grids = kwargs
    #print(grids)
    #print(grids.shape)
    grids = grids.reshape(11,2)
    pole_platform, plat_array = poles_platform_distance_constraint(grids,w_,garden_index)
    pole_pole, pole_array = all_poles_dist_constraint(grids)
    counts, _ = count_constraint(grids,w_,garden_index)
    m, n = pole_array.shape
    out = np.arange(m)[:, None] != np.arange(n)
    #violating_poles
    #print(plat_array)
    violating_poles_platforms = np.unique(np.argwhere(plat_array>0))
    violating_poles = np.unique(np.argwhere(pole_array[out]>0))
    violaters = np.union1d(violating_poles_platforms,violating_poles)
    all_ind = [i for i in range(11)]
    good_indices = np.setdiff1d(all_ind,violaters)
    #print(good_indices.shape)
    #print(good_indices)
    counts_satisfied = counts <= 0
    return max(counts,pole_pole,pole_platform), good_indices, counts_satisfied
"""

def in_polygon_constraint(point,poly):
    if poly.contains(point):
        return 0
    else:
        return 1


def all_constraints(w_, garden_index, grids,platform_polygon=None,platform_index=None):
    # if return value <= 0 then all constraints are satisfied
    nn = int(grids.size / 2)
    grids = grids.reshape(int(grids.size / 2), 2)
    pole_platform, plat_array = poles_platform_distance_constraint(grids, w_, garden_index)
    pole_pole, pole_array = all_poles_dist_constraint(grids)
    in_constraint = 0
    if platform_polygon is not None:
        for i in range(3):
            p = Point(grids[i,:])
            in_constraint += in_polygon_constraint(p,platform_polygon)
    """
    counts, _ = count_constraint(grids, w_, garden_index)
    m, n = pole_array.shape
    out = np.arange(m)[:, None] != np.arange(n)

    violating_poles_platforms = np.unique(np.argwhere(plat_array > 0))
    violating_poles = np.unique(np.argwhere(pole_array[out] > 0))
    violaters = np.union1d(violating_poles_platforms, violating_poles)
    all_ind = [i for i in range(nn)]
    best_ind = []
    good_indices = np.setdiff1d(all_ind, violaters)
    # check which platform they belong to and if count >= 5
    if good_indices.size > 0:
        count_good, constraint_list = count_constraint(grids[good_indices, :], w_, garden_index)
        it = iter(constraint_list)
        for x, ind in zip(it, good_indices):
            if next(it) <= 0:
                best_ind.append(ind)
    """
    return in_constraint,pole_pole, pole_platform


# https://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
def get_random_point_in_polygon(poly):
    x, y = poly.exterior.xy
    #plt.plot(x, y)
    #plt.show()
    #points = np.empty((x.shape[0],2))
    #points[:,0] = x
    #points[:,1] = y
    #idx = np.random.randint(x.shape[0], size=2)

    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p
    #return points[idx,:]



def random_search_in_platform(platform_ind, w_, garden_index, grids):
    u = 8
    v = 5
    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))
    if w_.shape[1] == ((u + 3) * v - 2):
        w_ = ut.convert_toxyz(w_, v, u, w_.shape[0])
    if grids.size <= 2:
        grids = grids.reshape(1, 2)
    else:
        grids = grids.reshape(int(grids.size / 2), 2)
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)
    for platform_index in platform_ind:
        ind_use = np.arange(u * 2) + platform_index * u * 2
        points = w_[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        count, _ = count_inside(w_, v, u, platform_index, garden_index, grids=grids)
        while count < 3:
            print("COUNT IN EXTERNAL LOOP {}".format(count))
            rand_point = np.array(get_random_point_in_polygon(poly)).reshape(1, 2)
            print(rand_point)
            tmp = grids
            tmp = np.append(tmp, rand_point, axis=0)
            pole_platform, _ = poles_platform_distance_constraint(tmp, w_, garden_index)
            pole_pole, _ = all_poles_dist_constraint(tmp)
            c = max(pole_pole, pole_platform)
            if c <= 0:
                count += 1
                grids = np.append(grids, rand_point, axis=0)
    return grids


def bo_in_platform(platform_ind, w_, garden_index, grids):
    u = 8
    v = 5
    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))
    if w_.shape[1] == ((u + 3) * v - 2):
        w_ = ut.convert_toxyz(w_, v, u, w_.shape[0])
    if grids.size <= 2:
        grids = grids.reshape(1, 2)
    else:
        grids = grids.reshape(int(grids.size / 2), 2)
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)

    grids = np.random.uniform(low=0, high=1, size=(15, 2))
    for platform_index in platform_ind:
        ind_use = np.arange(u * 2) + platform_index * u * 2
        points = w_[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)

        minx, miny, maxx, maxy = poly.bounds
        bounds = [
            {'name': 'p1_1', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p1_2', 'type': 'continuous', 'domain': (miny, maxy)},
            {'name': 'p2_1', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p2_2', 'type': 'continuous', 'domain': (miny, maxy)},
            {'name': 'p3_1', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p3_2', 'type': 'continuous', 'domain': (miny, maxy)}
        ]
        poles = np.random.uniform(low=minx, high=maxx, size=(3, 2))
        _, _, Y_init = all_constraints(w_, garden_index, poles, final_check=True)
        poles = poles.reshape(1, 6)
        Y_step = np.empty((1, 1))
        Y_step[0, 0] = Y_init

        count = 0
        context = {}
        while count < 3:
            bo_step = BayesianOptimization(f=None, model="GP_MCMC", domain=bounds, X=poles, Y=Y_step,
                                           maximize=False, acquisition="LCB")
            print("COUNT IN EXTERNAL LOOP {} for platform {}".format(count, platform_index))

            x_next = bo_step.suggest_next_locations()
            y_next, good_poles, counts = all_constraints(w_, garden_index, poles, final_check=True)
            print("good poles {}".format(good_poles))
            print("context {}".format(context))
            num_fixed_poles = len(context)
            print("num fixed poles {}".format(num_fixed_poles))
            # good_indices = list(chain.from_iterable((i, i + 1) for i in good_poles))
            fixed_poles = poles[np.argmin(Y_step), :]  # [good_indices]
            for p in good_poles:
                context["p{}_{}".format(p, 1)] = fixed_poles[p]
                context["p{}_{}".format(p, 2)] = fixed_poles[p + 1]
                count += 1
            y_tmp = np.empty((1, 1))
            y_tmp[0, 0] = y_next
            poles = np.vstack((poles, x_next))
            Y_step = np.vstack((Y_step, y_tmp))
            loss = y_tmp[0, 0]
            if loss <= 0:
                count += 1

        results = poles[np.argmin(Y_step), :].reshape(3, 2)
        grids[platform_index * 3:platform_index * 3 + 3, :] = results

    return grids



def bo_constraint_satisfaction(initial_grids, w_, bounds, garden_index, tol=0.5, maximize=False, acquisition="LCB",
                               model="GP_MCMC"):
    results = np.empty((11, 2))
    gp_results = np.empty((11, 2))

    _, _, Y_init = all_constraints(w_, garden_index, initial_grids)
    loss = 1000
    current_iter = 0
    X_step = initial_grids.reshape(1, 22)
    Y_step = np.empty((1, 1))
    Y_step[0, 0] = Y_init

    context = {}
    num_fixed_poles = 0
    fixed_ind = set()
    counts_satisfied = False
    final_grid = np.array(list(context.values()))
    while not counts_satisfied:
        print("ITERATION {}".format(current_iter))
        print("LOSS {}".format(loss))

        bo_step = BayesianOptimization(f=None, model=model, domain=bounds, X=X_step, Y=Y_step,
                                       maximize=maximize, acquisition=acquisition)
        try:
            x_next = bo_step.suggest_next_locations(context=context)
        except:
            x_next = bo_step.suggest_next_locations()
        y_next, good_poles, counts = all_constraints(w_, garden_index, x_next)

        print("good poles {}".format(good_poles))
        print("context {}".format(context))
        num_fixed_poles = len(context)
        print("num fixed poles {}".format(num_fixed_poles))
        # good_indices = list(chain.from_iterable((i, i + 1) for i in good_poles))
        fixed_poles = X_step[np.argmin(Y_step), :]  # [good_indices]
        for p in good_poles:
            fixed_ind.add(p)
            context["p{}_{}".format(p, 1)] = fixed_poles[p]
            context["p{}_{}".format(p, 2)] = fixed_poles[p + 1]
        if len(context) >= 2:
            counts, cl = count_constraint(x_next, w_, garden_index)
            num_not_good = len(cl[cl > 0])
            ind_platform = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
            # if num_not_good <= 4:
            #    platform_not_sat,_ = np.where(cl.reshape(5,2) > 0)
            #    # search in those
            #    # if found, break
            #    print("DOING RANDOM SEARCH IN PLATFORM")
            #    # do new bayesian optimization, stop the previous one
            #    results = search_in_platform(platform_not_sat,w_,garden_index,grids=x_next)
            #    break

            counts_satisfied = counts <= 0
            print("Are the counts satisfied? {}".format(counts_satisfied))

        y_tmp = np.empty((1, 1))
        y_tmp[0, 0] = y_next
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_tmp))
        loss = y_tmp[0, 0]
        current_iter += 1
        if counts_satisfied:
            # results = x_next.reshape(11,2)
            final_grid = np.array(list(context.values()))
        # print("SMALLEST LOSS {} ".format(np.min(Y_step)))
        # print("OPT PARAMS {}".format(X_step[np.argmin(Y_step), :]))
    results = X_step[np.argmin(Y_step), :].reshape(11, 2)

    opt_ind = bo_step.model.predict(bo_step.X)[0].argmin()
    gp_results = bo_step.X[opt_ind, :].reshape(11, 2)
    # print(bo_step.X[opt_ind, :])
    return results, gp_results, fixed_ind, final_grid


def all_constraints_online(w_, garden_index, grids):
    # if return value <= 0 then all constraints are satisfied
    nn = int(grids.size / 2)
    grids = grids.reshape(int(grids.size / 2), 2)
    pole_platform, plat_array = poles_platform_distance_constraint(grids, w_, garden_index)
    pole_pole, pole_array = all_poles_dist_constraint(grids)
    counts, _ = count_constraint(grids, w_, garden_index)
    m, n = pole_array.shape
    out = np.arange(m)[:, None] != np.arange(n)
    grid_red = grids
    print("GRIDS SHAPE {}".format(grids.shape))

    violating_poles_platforms = np.unique(np.argwhere(plat_array > 0))
    violating_poles = np.unique(np.argwhere(pole_array[out] > 0))
    violaters = np.union1d(violating_poles_platforms, violating_poles)
    all_ind = [i for i in range(nn)]
    good_indices = np.setdiff1d(all_ind, violaters)
    if grids.shape[0] > 1:
        grid_red = grids[good_indices, :]

    return max(counts, pole_pole, pole_platform), grid_red, counts


def get_polygon(w_, garden_index):
    # returns five polygons in a list
    u = 8
    v = 5
    poly_list = []
    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))
    if w_.shape[1] == ((u + 3) * v - 2):
        w_ = ut.convert_toxyz(w_, v, u, w_.shape[0])

    for platform_index in range(5):
        ind_use = np.arange(u * 2) + platform_index * u * 2
        points = w_[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        #plt.plot(points[:,0],points[:,1])
        #plt.show()
        poly = Polygon(points)
        poly_list.append(poly)

    return poly_list


def init_three_per_platform(w_, garden_index):
    # returns an array of dim [15,2]

    poly_list = get_polygon(w_, garden_index)
    ret_grids = np.zeros((15, 2))
    for i, polygon in enumerate(poly_list):
        p1 = np.array(get_random_point_in_polygon(polygon)).reshape(1, 2)
        p2 = np.array(get_random_point_in_polygon(polygon)).reshape(1, 2)
        p3 = np.array(get_random_point_in_polygon(polygon)).reshape(1, 2)
        ret_grids[i*3, :] = p1
        ret_grids[i*3 + 1, :] = p2
        ret_grids[i*3 + 2, :] = p3

    return ret_grids


def three_per_platform(w_, garden_index, tol=0.0, maximize=False, acquisition="LCB", model="GP_MCMC"):
    grids = init_three_per_platform(w_, garden_index)
    poly_list = get_polygon(w_, garden_index)

    # do one platform at a time, only have three (six) variables
    for platform_index in range(5):
        polygon = poly_list[platform_index]

        _,Y_init,_ = all_constraints(w_, garden_index, grids[platform_index * 3:platform_index * 3 + 3, :],
                                       platform_polygon=polygon,platform_index=platform_index)
        loss = 1000
        current_iter = 0
        X_step = np.zeros((1,6))

        Y_step = np.empty((1, 1))
        Y_step[0, 0] = Y_init
        minx, miny, maxx, maxy = polygon.bounds
        in_constraint = 4
        pole_2 = 5
        bounds = [
            {'name': 'p0x', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p0y', 'type': 'continuous', 'domain': (miny, maxy)},
            {'name': 'p1x', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p1y', 'type': 'continuous', 'domain': (miny, maxy)},
            {'name': 'p2x', 'type': 'continuous', 'domain': (minx, maxx)},
            {'name': 'p2y', 'type': 'continuous', 'domain': (miny, maxy)},
        ]

        while loss > tol or in_constraint > 0 or pole_2 > 0:
            print("ITERATION {} for platform {} has loss {}".format(current_iter, platform_index, loss))
            print("IN_CONSTRINT {}".format(in_constraint))
            print("POLE_2POLE {}".format(pole_2))
            bo_step = BayesianOptimization(f=None, model=model, domain=bounds, X=X_step, Y=Y_step,
                                           maximize=maximize, acquisition=acquisition)

            x_next = bo_step.suggest_next_locations()
            grids[platform_index * 3:platform_index * 3 + 3, :] = x_next.reshape(-1, 2)
            in_constraint,pole_2,pole_plat = all_constraints(w_, garden_index, grids[platform_index * 3:platform_index * 3 + 3, :],
                                           platform_polygon=polygon,platform_index=platform_index)
            y_next = max(in_constraint,pole_2)
            y_tmp = np.empty((1, 1))
            y_tmp[0, 0] = y_next
            X_step = np.vstack((X_step, x_next))
            Y_step = np.vstack((Y_step, y_tmp))
            loss = y_next
            current_iter += 1


        grids[platform_index * 3:platform_index * 3 + 3, :] = X_step[-1, :].reshape(3, 2)

    # go through grid, make the distance to platform constraint satisfied
    for i in range(grids.shape[0]):

        _, _, Y_init = all_constraints(w_, garden_index, grids[i, :])
        loss = 1000
        current_iter = 0
        X_step = np.zeros((1, 2))

        Y_step = np.empty((1, 1))
        Y_step[0, 0] = Y_init
        minx, miny, maxx, maxy = polygon.bounds
        mx = grids[i,0]
        my = grids[i,1]
        in_constraint = 4
        pole_plat = 5
        pole_2 = 5
        bounds = [
            {'name': 'p0x', 'type': 'continuous', 'domain': (mx-0.8, mx+0.8)},
            {'name': 'p0y', 'type': 'continuous', 'domain': (my-0.8, my+0.8)},
        ]
        while loss > 0.0 or pole_2 > 0:
            print("ITERATION {} for pole {} has loss {}".format(current_iter, i, loss))

            print("POLE_plat {}".format(pole_plat))
            bo_step = BayesianOptimization(f=None, model=model, domain=bounds, X=X_step, Y=Y_step,
                                           maximize=maximize, acquisition=acquisition)

            x_next = bo_step.suggest_next_locations()
            grids[i, :] = x_next.reshape(1, 2)
            _,pole_2,pole_plat = all_constraints(w_, garden_index, grids[i, :])
            y_next = max(pole_plat,pole_2)
            y_tmp = np.empty((1, 1))
            y_tmp[0, 0] = y_next
            X_step = np.vstack((X_step, x_next))
            Y_step = np.vstack((Y_step, y_tmp))
            loss = y_tmp[0,0]
            current_iter += 1

        grids[i, :] = X_step[-1, :].reshape(1, 2)


    return grids


def online_bo(initial_grids, w_, bounds, garden_index, tol=0.5, maximize=False, acquisition="LCB", model="GP_MCMC"):
    _, _, Y_init = all_constraints(w_, garden_index, initial_grids)
    loss = 1000
    current_iter = 0
    X_step = initial_grids.reshape(1, initial_grids.size)
    Y_step = np.empty((1, 1))
    Y_step[0, 0] = Y_init
    grids = initial_grids

    counts_satisfied = False
    while not counts_satisfied:
        print("ITERATION {}".format(current_iter))
        print("LOSS {}".format(loss))

        bo_step = BayesianOptimization(f=None, model=model, domain=bounds, X=X_step, Y=Y_step,
                                       maximize=maximize, acquisition=acquisition)

        x_next = bo_step.suggest_next_locations()
        grids = np.append(grids, x_next.reshape(1, 2), axis=0)
        y_next, grids, counts = all_constraints_online(w_, garden_index, grids)
        counts_satisfied = counts <= 0
        y_tmp = np.empty((1, 1))
        y_tmp[0, 0] = y_next
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_tmp))
        loss = y_tmp[0, 0]
        current_iter += 1

    return grids


# have to do it in 3d




def main():
    w_train_red, x_ground_truth, _ = mpb.load_raw_data(list_attrib)
    print(w_train_red.shape)
    grids = load_grids()
    garden_index = 0
    # initial_grids = np.random.normal(size=(1,2))
    # bound1 = [{'name': 'p', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2}]
    # grids = online_bo(initial_grids, w_train_red, bound1, garden_index)
    # plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
    #                    list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
    #                    samp_to_plot=0, grids=grids)

    res_grids = three_per_platform(w_train_red, garden_index)
    #res_grids = init_three_per_platform(w_train_red,garden_index)
    plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                        samp_to_plot=garden_index, grids=res_grids.reshape(15, 2))
    """
    bounds = [
              {'name': 'p0', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p1', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p2', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p3', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p4', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p5', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p6', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p7', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p8', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p9', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2},
              {'name': 'p10', 'type': 'continuous', 'domain': (-6, 6),"dimensionality":2}
              ]
    """
    # results, gp_results, fixed_ind, final_grid = bo_constraint_satisfaction(grids, w_train_red, bounds, garden_index)
    # fixed_ind = list(fixed_ind)
    # print(final_grid.shape)
    # final_grid = final_grid.reshape(len(fixed_ind),2)
    # final_loss, good_poles, _ = all_constraints(w_train_red, garden_index, final_grid,final_check=True)
    # print("final_loss {}".format(final_loss))

    # print("DONE")
    # print("FIXED IND {}".format(fixed_ind))
    # plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
    #                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
    #                        samp_to_plot=0, grids=results.reshape(11,2),pole_ind=np.array(fixed_ind))
    # plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
    #                   list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
    #                   samp_to_plot=0, grids=final_grid)

    # print(results)
    # bo_simple(grids, w_train_red, bounds, garden_index, tol=0.0, maximize=False, acquisition="LCB", model="GP_MCMC")

    # final_grids = bo_in_platform([0,1,2,3,4],w_train_red,garden_index,grids)
    # plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
    #                   list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
    #                    samp_to_plot=0, grids=final_grids)
    """
    fix the grid, only move the center such that the distances are good
    """

if __name__ == "__main__":
    main()
