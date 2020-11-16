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

from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, LinearRing
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

def give_linear_rings(w_,v,u,garden_index):
    if np.ndim(w_) == 1:
        w_ = w_.reshape((1, -1))

    if w_.shape[1] == ((u + 3) * v - 2):
        w_geom = ut.convert_toxyz(w_, v, u, w_.shape[0])

    points_list = []
    for i in range(v):
        ind_use = np.arange(u * 2) + i * u * 2
        points = w_geom[garden_index, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = LinearRing(points)
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

def plot_line(ax, ob, color):
    x, y = ob.xy
    ax.plot(x, y, color=color, alpha=0.7, linewidth=3,
            solid_capstyle='round', zorder=2)

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



def boundary_list(platforms,dist_to_platform):
    boundaries_in = []
    boundaries_out = []

    for i in range(len(platforms)):
        #bl = platforms[i].parallel_offset(distance=dist_to_platform,side="left")
        #br = platforms[i].parallel_offset(distance=dist_to_platform,side="right")
        #bl = platforms[i].difference(bl)
        #br = platforms[i].difference(br)
        #plt.plot(*Polygon(br).exterior.xy)
        #plt.show()
        bl = platforms[i].parallel_offset(distance=dist_to_platform,side="left")
        br = platforms[i].parallel_offset(distance=dist_to_platform,side="right")
        #boundaries_in.append(bl)
        #boundaries_out.append(br)

        bl = Polygon(bl)
        br = Polygon(br)

        outer_band = bl.difference(platforms[i])
        inner_band = platforms[i].difference(br)

        boundaries_in.append(inner_band)
        boundaries_out.append(outer_band)

    return boundaries_in, boundaries_out

def allowed_region(platform,boundaries_in,boundaries_out,index):

    platform_t = boundaries_out[index]#platform.difference(boundaries_in[index])
    #plt.plot(*platform_t.exterior.xy)
    # plt.plot(*boundaries_in[index].exterior.xy,c="red")
    #plt.plot(*boundaries_out[index].exterior.xy,c="yellow")

    #plt.plot(*platform.exterior.xy, c="green")
    #plt.show()

    if index == 0:
        return platform_t
    else:

        for i in (range(index)):
            # commenting the first one converged
            # commenting the second one did not
            #platform_t = platform_t.difference(boundaries_out[i])
            platform_t = platform_t.difference(boundaries_in[i])



    return platform_t

def count_loss(grids,platform):
    c=0
    for i in range(grids.shape[0]):
        p = Point(grids[i,:])
        if p.within(platform):
            c+=1
    #c = grids.shape[0]
    c3 = 3-c
    c5 = c-5
    return max(c3,c5)

def inside_loss(grids,region):
    l = 0
    for i in range(grids.shape[0]):
        p = Point(grids[i,:])
        if not p.within(region):
            l += 1
    return l

def distance_loss(grids):
    ret,_ = all_poles_dist_constraint(grids)
    return ret

def grid_for_platform(platform, bounds,grids,initial):
    #grids = np.empty((1,2))
    X_step = np.zeros((1, 2))
    Y_init = 3
    Y_step = np.empty((1, 1))
    Y_step[0, 0] = Y_init
    loss = 1000
    current_iter = 0
    while loss > 0.0:
        bo_step = BayesianOptimization(f=None, model="GP", domain=bounds, X=X_step, Y=Y_step,
                                       maximize=False, acquisition="LCB")

        x_next = bo_step.suggest_next_locations()
        #print("ITERATION {}".format(grids.shape))
        tmp_point = Point(x_next[0])
        if initial and tmp_point.within(platform):
            grids = x_next.reshape(1, 2)
            current_iter += 1
            initial = False
        elif tmp_point.within(platform):
            tmp_grid = np.append(grids, x_next.reshape(1, 2), axis=0)
            distances = np.linalg.norm(tmp_grid - tmp_grid[:,None], axis=-1)
            mask = np.where(~np.eye(distances.shape[0], dtype=bool)==True)
            if np.all(distances[mask] >= 1.9):
                grids = np.append(grids,x_next.reshape(1, 2), axis=0)
                loss = count_loss(grids,platform)
                #print("COUNT LOSS {}".format(loss))
            else:
                loss = max(count_loss(grids,platform) , (2-np.amin(distances[mask])))
                #print("LOSS {}".format(loss))
        else:
            loss = 1000
        if current_iter>=1:
            current_iter += 1

        #print(grids.shape)

    return grids


def fix_poles(platforms,linear_rings):
    grid = np.empty((1,2))
    boundaries_in, boundaries_out = boundary_list(linear_rings,0.8)
    initial_flag = False
    poles_per_platform = np.zeros((5))
    for i, platform in enumerate(reversed(platforms)):

        print("PLATFORM {}".format(i))
        region = allowed_region(platform,boundaries_in,boundaries_out,4-i)

        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        """
        try:
            for geom in region.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                #fig.plot(*platform.exterior.xy,c="green")
            axs.fill(*platform.exterior.xy, alpha=0.5, fc='g', ec='none')
                #plt.show()
        except:
            plt.plot(*region.exterior.xy)
            plt.plot(*platform.exterior.xy,c="green")
            plt.show()
        """
        # now pick between 3 and 5 points in region s.t they have a min distance of 2
        xmin, ymin, xmax, ymax = region.bounds
        bounds = [
            {'name': 'px', 'type': 'continuous', 'domain': (xmin, xmax)},
            {'name': 'py', 'type': 'continuous', 'domain': (ymin, ymax)}
        ]
        if i == 0:
            initial_flag = True
        else:
            initial_flag = False

        grid = grid_for_platform(region,bounds,grid,initial_flag)
        num_poles = grid.shape[0]
        if i == 0:
            poles_per_platform[i] = num_poles
        else:
            poles_per_platform[i] = num_poles - poles_per_platform[i-1]
        #print(grids_tmp.shape)
        #grid = np.append(grid,grids_tmp,axis=0)
        #print(grid.shape)
        # project tmp grid to other felan

        #plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
        #                    list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
        #                    samp_to_plot=0, grids=grid)
    return grid, poles_per_platform



def main():

    #w_train_red, x_ground_truth, _ = mpb.load_raw_data(list_attrib)
    #grids = load_grids()
    garden_index = 0

    platforms = give_polygons(w_train_red,5,8,garden_index)
    linear_rings = give_linear_rings(w_train_red,5,8,garden_index)
    grids, poles_index = fix_poles(platforms,linear_rings)
    poles_index = poles_index.astype(int)
    plot_platform_grids(w_train_red, x_ground_truth, v=5, u=8,
                                            list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                                           samp_to_plot=0, grids=grids,pole_ind=poles_index)
    print(grids)



if __name__ == "__main__":
    main()
