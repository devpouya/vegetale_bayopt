import numpy as np
import os
import pickle as pk

import utils as ut
import move_platforms_bo as mpb
from platypus import NSGAII, Problem, Real

import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
from GPyOpt.methods import BayesianOptimization

from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.affinity import translate, rotate

curr_dir_repo = os.getcwd().split('/src/python')[0]
dataset_dir = curr_dir_repo + '/data/'  # + f_experiment
results_dir = curr_dir_repo + '/results/'  # + f_experiment

list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

w_train_red, x_ground_truth, _ = mpb.load_raw_data(list_attrib)



def load_grids(grid_name_string='grid_points_coord.pkl'):
    gridfile = os.path.join(dataset_dir + "/grid/", grid_name_string)
    grids = pk.load(open(gridfile, 'rb'))
    return grids

grids = load_grids()

def give_polygons(w_,v,u,garden_index):

    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))

    if w_.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_, v, u, w_.shape[0])

    points_list = []
    for i in range(v):
        ind_use = np.arange(u * 2) + i * u * 2
        points = w_geom[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        points_list.append(poly)
    return points_list


def calc_distance_to_platform(w_geom, v, u, garden_index, w_means=None, w_std=None,
                              grids=None):
    mpt_g = convex_hull_garden(w_geom,garden_index)

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
    pole_in_one_poly = np.zeros((grids.shape[0]))
    for iw in range(v):
        ind_use = np.arange(u * 2) + iw * u * 2
        points = w_geom[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)

        for pole_index in range(grids.shape[0]):
            pole = Point(grids[pole_index, :])
            if pole.within(mpt_g.convex_hull):
                if pole.within(poly):
                    dist = poly.exterior.distance(pole)
                    distances_matrix[pole_index, iw] = dist
                    inside_dist.append(dist)
                    pole_in_one_poly[pole_index] = 1
                else:
                    dist = poly.boundary.distance(pole)
                    distances_matrix[pole_index, iw] = dist
                    outside_dist.append(dist)
            else:
                distances_matrix[pole_index, iw] = 5
    count = 0
    loss = 0
    for pole_index in range(grids.shape[0]):
        if pole_in_one_poly[pole_index] == 0:
            count += 1
    if count >= 5:
        loss += count*0.1

    return distances_matrix, np.array(inside_dist), np.array(outside_dist), loss


def count_inside(w_geom, v, u, garden_index, w_means=None, w_std=None,
                 grids=None):
    count = 0
    inside_index = []
    h_max = 12
    # sort platforms by height
    platforms = w_geom[garden_index, :]
    h_index = [50, 51, 52]
    heights = platforms[h_index]
    sorted_args = np.argsort(heights)
    sorted_args += 1
    sorted_indices = np.zeros(5)
    sorted_indices[0] = 0
    sorted_indices[4] = 4
    sorted_indices[1:4] = sorted_args
    pole_heights = np.zeros(grids.shape[0])
    count_per_platform = np.zeros(5)

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


    for sorted_plat in sorted_indices:
        sorted_plat = int(sorted_plat)
        ind_use = np.arange(u * 2) + sorted_plat * u * 2
        points = w_geom[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        for i in range(grids.shape[0]):
            pole = Point(grids[i, :])
            #if pole_heights[i] <= h_max:
            if pole.within(poly):
                count_per_platform[sorted_plat] += 1
                if sorted_plat == 0:
                    pole_heights[i] = 4.5
                elif sorted_plat == 4:
                    pole_heights[i] = 12
                else:
                    index = sorted_plat - 1
                    pole_heights[i] = heights[index]

    return count_per_platform


def pole_to_pole_distances(grids):
    if grids.ndim < 2:
        grids = grids.reshape(grids.shape[0] / 11, 2)

    if grids.ndim < 2:
        grids = grids.reshape(11, 2)

    pole_to_pole = np.zeros((grids.shape[0], grids.shape[0]))

    for i in range(grids.shape[0]):
        for j in range(grids.shape[0]):
            dist = np.linalg.norm(grids[i, :] - grids[j, :])
            pole_to_pole[i, j] = dist

    return pole_to_pole


def all_poles_dist(grids):
    if grids.ndim < 2:
        grids = grids.reshape(11, 2)
    b = grids.reshape(grids.shape[0], 1, grids.shape[1])

    return np.sqrt(np.einsum('ijk, ijk->ij', grids - b, grids - b))


def count_constraint(grids, w_, garden_index, context=None):
    # each platform should have between 3 and 5 poles
    # define constraint s.t it is satisfied if f(x) <= 0
    # a list of constraints, each constraint is a number
    # if all numbers are <= 0, then all constraints are fulfilled
    if context is not None:
        grids = np.array(list(context.values()))
        # print(grids.shape)
        # grids = grids.reshape(grids.shape[0] / 2, 2)

    """
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
    """
    count_per_platform = count_inside(w_,v=5,u=8,garden_index=garden_index,grids=grids)
    smaller_than = 3 - count_per_platform
    bigger_than = count_per_platform - 5
    all = np.array([bigger_than,smaller_than])
    all = normalize(all)
    #stc = np.amax(smaller_than)
    #btc = np.amax(bigger_than)
    #max_f = max(stc,btc)
    max_f = np.amax(all)
    # constraint_list will be of size 10, each two elements correspond to one platform
    return max_f

def convex_hull_garden(w_,garden_index,v=5,u=8):
    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))

    if w_.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_, v, u, w_.shape[0])

    points_list = []
    for i in range(v):
        ind_use = np.arange(u * 2) + i * u * 2
        points = w_geom[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        points_list.append(poly)
    mtp = MultiPolygon(points_list)
    #cvx = mtp.convex_hull
    return mtp


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
    distances, inside_dist, outside_dist,loss = calc_distance_to_platform(w_, v=5, u=8, garden_index=garden_index,
                                                                     grids=grids)
    #print(distances)
    # inside_constraint = 0.8-inside_dist
    # outside_constraint = 0.8-outside_dist
    # inside_max = np.amax(inside_constraint)
    # outside_max = np.amax(outside_constraint)
    #constraints = np.sum(distances < 0.6)
    #sat = np.all(distances >= 0.6,axis=1)
    #good_ind = np.argwhere(sat == True)
    constraints = 0.8 - distances
    #print(constraints.shape)
    constraints = normalize(constraints)
    c_max = np.amax(constraints)
    if loss > 0:
        c_max += loss

    return c_max

def s_lambda(x1,x2):
    lamb = np.random.normal(loc=2,size=1)
    #y = (np.any(0,x/lamb))**2
    xx1 = max(x1/lamb,0)
    xx2 = max(x2/lamb,0)
    res = min(xx1,xx2)**2
    return res #np.amin(np.where(x/lamb > 0))**2

def plot_platform_grids(w_train_red, x_vals, v=5, u=8,
                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                        samp_to_plot=0, grids=None, pole_ind=None):
    w_ = w_train_red[0, :].reshape(1, 53)
    ut.plot_contours(w_, v=5, u=8, x_vals=x_vals,
                     list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                     samp_to_plot=0, grids=grids, pole_ind=pole_ind)


def in_polygon(point, poly):
    if poly.contains(point):
        return 1
    else:
        return 0


def all_constraints(w_, garden_index, grids, platform_polygon=None, platform_index=None):
    # if return value <= 0 then all constraints are satisfied
    grids = grids.reshape(int(grids.size / 2), 2)
    pole_platform = poles_platform_distance_constraint(grids, w_, garden_index)

    #pole_pole, _ = all_poles_dist_constraint(grids)

    cc = count_constraint(grids,w_,garden_index)
    #cc /= 5
    #print("pp: {}".format(pole_platform))
    #print("cc: {}".format(cc))
    final= max(pole_platform,cc)

    return final,pole_platform, cc




def move_grid_by_center(new_center, grids, prev_center):
    points_list = []
    for i in range(grids.shape[0]):
        p = Point(grids[i, :])
        points_list.append(p)

    multi_point = MultiPoint(points_list)
    move_x = prev_center[0,0]-new_center[0,0]
    move_y = prev_center[0,1]-new_center[0,1]
    translated = translate(multi_point, xoff=move_x, yoff=move_y)
    rotated = rotate(translated,angle=new_center[0,2])
    ret_grids = np.zeros_like(grids)
    for i, p in enumerate(rotated):
        ret_grids[i, 0] = p.xy[0][0]
        ret_grids[i, 1] = p.xy[1][0]
    return ret_grids



def grid_bo(w_, garden_index, grids, fit_counts=True):
    # first compute convex hull of grids
    # parametrize problem by just the center of the grid
    points_list = []
    for i in range(grids.shape[0]):
        p = Point(grids[i, :])
        points_list.append(p)

    mtp = MultiPoint(points_list)
    cvx_hull = mtp.convex_hull
    grid_center = cvx_hull.centroid
    grid_center_x = grid_center.xy[0][0]
    grid_center_y = grid_center.xy[1][0]
    gminx, gminy, gmaxx, gmaxy = cvx_hull.bounds

    mtp_g = convex_hull_garden(w_,garden_index).convex_hull
    minx, miny, maxx, maxy = mtp_g.bounds


    bounds = [
        {'name': 'center_x', 'type': 'continuous', 'domain': (gminx, gmaxx)},
        {'name': 'center_y', 'type': 'continuous', 'domain': (gminy, gmaxy)},
        {'name': 'rotate', 'type': 'continuous', 'domain': (0, 180)}
    ]

    _,pp,cc = all_constraints(w_, garden_index, grids)
    #if fit_counts:
    #    Y_init = cc
    #    tol = 0.0
    #else:
    #    Y_init = pp
    #    tol = 0.1
    tol = 0.19
    Y_init = max(cc,pp)
    #Y_init = pp
    loss = 1000
    current_iter = 0
    X_step = np.zeros((1, 3))
    X_step[0, 0] = grid_center_x
    X_step[0, 1] = grid_center_y
    Y_step = np.empty((1, 1))
    Y_step[0, 0] = Y_init
    prev_center = X_step

    while loss > tol:
        if current_iter > 200:
            print("OVER THE LIMIT")
            opt_ind = bo_step.model.predict(bo_step.X)[0].argmin()
            x_opt = bo_step.X[opt_ind,:]
            x_opt = x_opt.reshape(1,3)

            x_opt = move_grid_by_center(x_opt, grids, prev_center)
            return x_opt

        bo_step = BayesianOptimization(f=None, model="GP", domain=bounds, X=X_step, Y=Y_step,
                                       maximize=False, acquisition="EI")

        x_next = bo_step.suggest_next_locations()

        grids = move_grid_by_center(x_next, grids,prev_center)
        prev_center = x_next
        _,pp,cc = all_constraints(w_, garden_index, grids)
        loss = max(pp,cc)
        #loss = pp
        #if fit_counts:
        #    loss = cc
        #else:
        #    loss = abs(pp)
        #try:
        #    scale = s_lambda(pp,cc)
        #    print(scale)
        #except:
        #    scale = 1
        #    print("EXCEPTION")
        #if pp == 0 and cc == 0:
        #    loss = 0
        #elif scale > 0:
        #    loss *=scale
        print("ITER {} LOSS {}".format(current_iter,loss))
        y_next = loss
        y_tmp = np.empty((1, 1))
        y_tmp[0, 0] = y_next
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_tmp))
        loss = y_next

        current_iter += 1
        #if current_iter == 50:
        #    opt_ind = bo_step.model.predict(bo_step.X)[0].argmin()
        #    x_opt = bo_step.X[opt_ind, :]
        #    grids = move_grid_by_center(x_opt, grids, prev_center)
        #    return grids

    return grids


def which_poles_in(w_,grids,garden_index,v=5,u=8):
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
    good_ind = []
    for pole in range(grids.shape[0]):
        point = Point(grids[pole,:])
        for i in range(v):
            ind_use = np.arange(u * 2) + i * u * 2
            points = w_[garden_index, ind_use].reshape(u, -1)
            points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
            poly = Polygon(points)
            is_in = in_polygon(point,poly)
            if is_in:
                good_ind.append(pole)
                break

    return good_ind

def platypus_obj(vars):
    w_ = w_train_red
    grids = load_grids()

    x = vars[0]
    y = vars[1]
    r = vars[2]
    points_list = []
    for i in range(grids.shape[0]):
        p = Point(grids[i, :])
        points_list.append(p)

    mtp = MultiPoint(points_list)
    cvx_hull = mtp.convex_hull
    grid_center = cvx_hull.centroid
    grid_c = np.zeros((1, 3))
    grid_c[0, 0] = grid_center.xy[0][0]
    grid_c[0, 1] = grid_center.xy[1][0]
    new_cent = np.zeros_like(grid_c)
    new_cent[0,0] = x
    new_cent[0,1] = y
    new_cent[0,2] = r
    grids = move_grid_by_center(new_cent,grids,grid_c)
    _, pp, cc = all_constraints(w_, 0, grids)
    print((cc,pp))
    return [cc,pp]


def platypus_grid(w_,garden_index,grids):
    # first compute convex hull of grids
    # parametrize problem by just the center of the grid
    points_list = []
    for i in range(grids.shape[0]):
        p = Point(grids[i, :])
        points_list.append(p)

    mtp = MultiPoint(points_list)
    cvx_hull = mtp.convex_hull
    gminx, gminy, gmaxx, gmaxy = cvx_hull.bounds



    problem = Problem(3,2)
    problem.types[0] = Real(gminx,gmaxx)
    problem.types[1] = Real(gminy,gmaxy)
    problem.types[2] = Real(0,180)
    problem.function = platypus_obj
    algorithm = NSGAII(problem)

    algorithm.run(100)
    print(algorithm.result)

    return algorithm.result


def break_grid_bo(w_, garden_index, grids):
    # first compute convex hull of grids
    # parametrize problem by just the center of the grid
    num_poles = grids.shape[0]
    bounds = []
    for i in range(num_poles):
        name_s = "p"+str(i)
        lower_x = grids[i,0] - 0.8
        upper_x = grids[i,0] + 0.8
        lower_y = grids[i,1] - 0.8
        upper_y = grids[i,1] + 0.8
        domainx = (lower_x,upper_x)
        domainy = (lower_y,upper_y)
        dx = {'name':name_s+"x",'type':'continuous','domain':domainx}
        dy = {'name':name_s+"y",'type':'continuous','domain':domainy}
        bounds.append(dx)
        bounds.append(dy)


    Y_init,pole_platform,cc ,_= all_constraints(w_,garden_index,grids)

    loss = 1000
    current_iter = 0
    X_step = np.zeros((1, num_poles*2))
    Y_step = np.empty((1, 1))
    Y_step[0, 0] = Y_init
    context = {}

    while loss > 2:
        if current_iter >= 200:
            return grids
        bo_step = BayesianOptimization(f=None, model="GP", domain=bounds, X=X_step, Y=Y_step,
                                       maximize=False, acquisition="LCB")
        try:
            x_next = bo_step.suggest_next_locations(context=context)
        except:
            x_next = bo_step.suggest_next_locations()
        grids = x_next.reshape(num_poles,2)
        loss,pole_platform,cc,good_ind = all_constraints(w_,garden_index,grids)
        for i in good_ind:
            context["p{}x".format(i)] = grids[i,0]
            context["p{}y".format(i)] = grids[i,1]
        print("Iteration {} has loss {}".format(current_iter, loss))
        print("POLE2PLAT {}".format(pole_platform))
        y_next = loss
        y_tmp = np.empty((1, 1))
        y_tmp[0, 0] = y_next
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_tmp))
        loss = y_next
        current_iter += 1

    return grids



def main():

    #w_train_red, x_ground_truth, _ = mpb.load_raw_data(list_attrib)
    grids = load_grids()
    garden_index = 0

    mpt_g = convex_hull_garden(w_train_red,garden_index).convex_hull
    center = mpt_g.centroid
    points_list = []
    for i in range(grids.shape[0]):
        p = Point(grids[i, :])
        points_list.append(p)

    mtp = MultiPoint(points_list)
    cvx_hull = mtp.convex_hull
    grid_center = cvx_hull.centroid
    grid_c = np.zeros((1,3))
    grid_c[0,0] = grid_center.xy[0][0]
    grid_c[0,1] = grid_center.xy[1][0]

    poly_c = np.zeros((1,3))
    poly_c[0,0] = center.xy[0][0]
    poly_c[0,1] = center.xy[1][0]
    #grid_center_x = grid_center.xy[0][0]
    #grid_center_y = grid_center.xy[1][0]
    minx, miny, maxx, maxy = mpt_g.bounds
    x = np.arange(minx, maxx, step=2)

    y = np.arange(miny, maxy, step=2)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    grids = np.array([xx, yy]).transpose()
    grids = move_grid_by_center(poly_c, grids, grid_c)








    good_ind = which_poles_in(w_train_red,grids,garden_index)
    grids = grids[good_ind,:]



    # just optimize the counts with this, then break grid
    res_grids = grid_bo(w_train_red, garden_index, grids)



    #final_grids = grid_bo(w_train_red, garden_index, grids,fit_counts=False)
    """
    solutions = platypus_grid(w_train_red,garden_index,grids)
    final_solution = solutions[-1]
    final_center = np.zeros_like(grid_c)
    final_center[0,0] = final_solution[0]
    final_center[0,1] = final_solution[1]
    final_center[0,2] = final_solution[2]
    final_grid = move_grid_by_center(final_center,grids,grid_c)
    """
    #good_ind = which_poles_in(w_train_red, final_grids, garden_index)
    #final_grids = final_grids[good_ind,:]
    #final_grids = grid_bo(w_train_red,garden_index,final_grids,fit_counts=False)
    #good_ind = which_poles_in(w_train_red, final_grids, garden_index)
    #final_grids = final_grids[good_ind,:]
    plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                        samp_to_plot=garden_index, grids=res_grids)
    #good_ind = which_poles_in(w_train_red, final_grids, garden_index)
    #final_grids = final_grids[good_ind,:]

    #good_ind = which_poles_in(w_train_red,res_grids,garden_index)
    #res_grids = res_grids[good_ind,:]
    #print("GRIDS FIXED")
    #print("NUM FIXED {}".format(res_grids.shape[0]))
    #final_grids = break_grid_bo(w_train_red,garden_index,res_grids)





if __name__ == "__main__":
    main()
