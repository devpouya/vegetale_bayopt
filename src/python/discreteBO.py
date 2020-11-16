import numpy as np
import os
import pickle as pk

import utils as ut
import move_platforms_bo as mpb
import  toptobottom
from platypus import NSGAII, Problem, Real
from itertools import chain
import scipy
import GPy
import GPyOpt
from GPyOpt.models import GPModel
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.core.evaluators import Sequential
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
from GPyOpt.methods import BayesianOptimization

from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, LinearRing
from shapely.affinity import translate, rotate

import generate_platform

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.LCB import AcquisitionLCB


"""

fix hexagonal grid, move platforms on grid one by one, until satisfies constraints,
have checkerboard acquisition
get the smalles island (loss)
fine tune in that island

10 dimensional grid of feasible and infeasible points

need to look for discrete position (10 dimensional 0-1 vector) with lowest loss
refine it in that regieme 

"""

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


def calc_distance_to_platform(w_geom, v, u, w_means=None, w_std=None,
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

    for isamp in [0]:
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


def count_inside(w_geom, v, u, platform_index, w_means=None, w_std=None,
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
    for isamp in [0]:
        ind_use = np.arange(u * 2) + platform_index * u * 2
        points = w_geom[isamp, ind_use].reshape(u, -1)
        points = np.concatenate([points, points[0, :].reshape(1, 2)], axis=0)
        poly = Polygon(points)
        for i in range(grids.shape[0]):
            pole = Point(grids[i, :])
            if pole.within(poly):
                count += 1
                inside_index.append(i)
    return count, inside_index


def poles_platform_distance_constraint(grids, w_):
    # if inside dist >= 0.8
    # if outside dist >= 0.8
    distances, inside_dist, outside_dist = calc_distance_to_platform(w_, v=5, u=8,
                                                                     grids=grids)
    constraints = 0.8 - distances
    c_max = np.amax(constraints)
    return c_max, constraints


def count_constraint(grids, w_, num_platforms=5):
    # each platform should have between 3 and 5 poles
    # define constraint s.t it is satisfied if f(x) <= 0
    # a list of constraints, each constraint is a number
    # if all numbers are <= 0, then all constraints are fulfilled
    constraint_list = []

    max_f = -100000
    for i in range(num_platforms):
        count, inside_index = count_inside(w_, v=5, u=8, platform_index=i, grids=grids)
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


def check_if_valid(platforms, supports):
    count_c, _ = count_constraint(supports, platforms)
    dist_c, _ = poles_platform_distance_constraint(supports, platforms)
    if count_c <= 0 and dist_c <= 0:
        is_valid = True
    else:
        is_valid = False

    return is_valid

def valid_loss(platforms,supports):
    count_c, _ = count_constraint(supports, platforms)
    dist_c, _ = poles_platform_distance_constraint(supports, platforms)
    return count_c + dist_c


def check_if_valid_x(x):
    #platforms[:,all_inds] = x.transpose()
    indicator = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        platforms[0,xy_ind] = x[i,:]
        if check_if_valid(platforms,supports):
            indicator[i] = 1
    return indicator

def make_grid(limit_high=6, limit_low=-6, num_points=3):
    grid_1d = np.linspace(limit_low, limit_high, num=num_points)
    grid = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, grid_1d, grid_1d,
                                grid_1d, grid_1d, grid_1d, grid_1d, grid_1d,
                                )).T.reshape(-1, 10)

    ret_grid = np.random.permutation(grid)
    return ret_grid


def compute_possible_positions(supports, platforms,grid):
    # xy = platforms[:,xy_ind]
    indicator = np.zeros((grid.shape[0]))
    for i in range(grid.shape[0]):
        platforms[:, xy_ind] = grid[i, :]
        is_valid = check_if_valid(platforms, supports)
        if is_valid:
            print("YEY")
            indicator[i] = 1
        else:
            print("NOO")
    print(len(np.where(indicator==1)))


    return grid, indicator

class constrained_LCB(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(self, model, constraint_model,space, optimizer=None, cost_withGradients=None,
                 num_samples=10, exploration_weight=40):
        self.model = model
        self.constraint_model = constraint_model
        self.optimizer = optimizer
        super(constrained_LCB,self).__init__(model,space,optimizer)
        self.num_samples = num_samples
        self.LCB = AcquisitionLCB(model,space,optimizer,cost_withGradients)
        self.exploration_weight = exploration_weight
        self.xall = np.empty((1,13))
        self.yall = np.empty((1,1))

    def acquisition_function(self,x):
        m,s = self.model.predict(x)
        f_acqu = -m + self.exploration_weight*s
        # need to integrate the indicator
        # have a GP model for the constraints
        #dist = scipy.stats.norm(loc=m,scale=s)
        #f_constraint = dist.cdf(x)#-indicator_m+self.exploration_weight*indicator_s
        indicator = check_if_valid_x(x)
        self.xall = np.vstack((self.xall, x))
        self.yall = np.concatenate((self.yall, indicator.reshape(indicator.shape[0],1)))
        self.constrain_model = self.constraint_model.updateModel(X_all=self.xall,Y_all=self.yall,X_new=x,Y_new=indicator)
        indicator_m,indicator_s = self.constraint_model.predict(x)
        indicator_acqu = -indicator_m+self.exploration_weight*indicator_s
        f_acqu = f_acqu.transpose()*indicator_acqu+np.random.beta(10,20,size=(x.shape[0]))
        print(f_acqu)
        return f_acqu


    def acquisition_function_withGradients(self, x):
        #try:
        #    m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        #except:
        m, s = self.model.predict(x)
        dmdx = np.random.normal(size=(m.shape[0],m.shape[0]))
        dsdx = np.random.normal(size=(m.shape[0],m.shape[0]))
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        f_constraint = 1#scipy.stats.norm.cdf(x, loc=m, scale=s)
        df_constraint = 1#scipy.stats.norm.cdf(x,loc=dmdx,scale=dsdx)
        #mc, sc= self.constraint_model.predict(x)
        #dmdxc, dsdxc = self.constraint_model.predictive_gradients(x)
        #f_acquc = -mc + self.exploration_weight * sc
        #df_acquc = -dmdxc + self.exploration_weight * dsdxc

        return f_acqu*f_constraint, df_acqu*df_constraint


from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from numpy.random import beta


class jitter_integrated_EI(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(self, model, space,constraint_model ,optimizer=None, cost_withGradients=None, par_a=1, par_b=1, num_samples=10):
        super(jitter_integrated_EI, self).__init__(model, space, optimizer)

        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        self.samples = beta(self.par_a, self.par_b, self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)
        self.xall = np.empty((1,13))
        self.yall = np.empty((1,1))
        self.constraint_model = constraint_model

    def acquisition_function(self, x):
        acqu_x = np.zeros((x.shape[0], 1))
        #x = np.matmul(x,np.random.normal(size=(13,1)))
        #print(x.shape)
        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]+np.random.normal(size=1)
            acqu_x += self.EI.acquisition_function(x)

        #indicator = check_if_valid_x(x)
        #self.xall = np.vstack((self.xall, x))
        #self.yall = np.concatenate((self.yall, indicator.reshape(indicator.shape[0], 1)))
        #self.constrain_model = self.constraint_model.updateModel(X_all=self.xall, Y_all=self.yall, X_new=x,
        #                                                        Y_new=indicator)
        #indicator_m, indicator_s = self.constraint_model.predict(x)
        #indicator_acqu = -indicator_m + self.exploration_weight * indicator_s
        #f_acqu = f_acqu.transpose() * indicator_acqu + np.random.beta(10, 20, size=(x.shape[0]))
        #print(f_acqu)
        #m,s = self.constraint_model.predict(x)
        #acqu_ind = -m+10*s
        #print(indicator.shape)
        #print(x.shape)
        acqu_x = acqu_x / self.num_samples
        #acqu_x = acqu_x.transpose()*np.random.normal(size=x.shape[0])
        #print("ACQUIsiition {}".format(acqu_x))
        #acqu_x = acqu_x.transpose()*acqu_ind
        return acqu_x

    def acquisition_function_withGradients(self, x):
        acqu_x = np.zeros((x.shape[0], 1))
        acqu_x_grad = np.zeros(x.shape)

        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]+np.random.normal(size=1)
            acqu_x_sample, acqu_x_grad_sample = self.EI.acquisition_function_withGradients(x)
            acqu_x += acqu_x_sample
            acqu_x_grad += acqu_x_grad_sample
        return acqu_x / self.num_samples, acqu_x_grad / self.num_samples

def black_box_function(index,x_,x_true,w_platforms,bottom_top_heights,output_attrib, params):

    w_ = w_platforms[index, :]
    w_ = w_.reshape(1, w_.shape[0])
    w_[0, xy_ind] = params.reshape(len(xy_ind), )
    x_in = x_[index, :]
    x_in = x_in.reshape(1, x_in.shape[0])
    x = mpb.GH_attr(w_, x_in, bottom_top_heights)

    xgh = x[output_attrib]
    xgh = np.asarray(xgh.values)
    print(xgh)
    print(x_true)
    print(xgh.shape)
    print(x_true.shape)
    #loss = mpb.abs_diff_loss(xgh, x_true,index)
    loss = np.empty((1,1))
    loss[0,0] = (abs(xgh[0,0] - x_true[0,0]) + abs(xgh[0,1] - x_true[0,1]))/2

    return loss, xgh, w_platforms

def run_bo_(w_platforms, x_train_red,x_true, bottom_top_heights,output_attrib,bounds,
            model,acquisition,
            evaluator=None,
            tol=10,supports=None):
    w_ = w_platforms
    x_ = x_train_red
    results = np.empty((1, len(xy_ind)))



    X_init = np.random.normal(size=(1, len(xy_ind)))
    Y_init, x_gh, w_ = black_box_function(0, x_,x_true, w_,bottom_top_heights,output_attrib, X_init)
    loss = Y_init[0,0]
    current_iter = 0
    X_step = X_init
    Y_step = Y_init

    min_loss = loss
    w_check = w_
    w_check[0, xy_ind] = X_init.reshape(len(xy_ind), )
    is_valid = check_if_valid(w_check, supports)



    while 12 < loss or not is_valid:

        print("TOL {} and LOSS {}".format(tol,loss))
        print("ITER {} ".format(current_iter))

        bo_step = GPyOpt.methods.ModularBayesianOptimization(model, bounds, None,
                                                             acquisition, evaluator,
                                                             Y_init=Y_step,X_init=X_step)



        x_next = bo_step.suggest_next_locations()

        #y_next, x_gh, w_ = black_box_function(0, x_,x_true, w_,bottom_top_heights,output_attrib,x_next)
        w_check = w_
        w_check[0, xy_ind] = x_next.reshape(len(xy_ind), )
        is_valid = check_if_valid(w_check, supports)
        lool = 0
        loss_v = 100
        while not is_valid:
            lool += 1
            #print(lool)
            bo_step = GPyOpt.methods.ModularBayesianOptimization(model, bounds, None,
                                                                 acquisition, evaluator,
                                                                 Y_init=Y_step, X_init=X_step)
            x_next = bo_step.suggest_next_locations()
            print(loss_v)
            w_check = w_
            w_check[0, xy_ind] = x_next.reshape(len(xy_ind), )
            loss_v = valid_loss(w_check, supports)
            y_next = np.array([[loss_v]])
            X_step = np.vstack((X_step, x_next))
            Y_step = np.vstack((Y_step, y_next))
            if lool >= 200:
                return x_next
            if loss_v <= 0:
                is_valid = True
                return x_next
        y_next, x_gh, w_ = black_box_function(0, x_,x_true, w_,bottom_top_heights,output_attrib,x_next)

        #if not is_valid:
        #   print("INVALID")
        #else:
        #    print("VALID")
        #    return x_next
        x_[0, :2] = x_gh[0, :]
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))

        loss = y_next[0, 0]
        min_loss = min(loss,min_loss)

        current_iter += 1
        if is_valid and loss < 12:
            return x_next
    print("SMALLEST LOSS {} ".format(np.min(Y_step)))
    results[0, :] = X_step[np.argmin(Y_step), :]

    return results









def main():

    # input the desired rain and sun occlusions

    rain_des =25 #float(input("Please enter the desired rain_occlusion: "))
    sun_des = 25#float(input("Please enter the desired sun_occlusion: "))

    x_true = np.array([[rain_des], [sun_des]])
    w_train_red, x_train_red, bottom_top_heights = mpb.load_raw_data(list_attrib)
    x_true = x_true.transpose()
    # load the 11 point grid
    global supports
    supports = load_grids()
    output_attrib = ['occlusion_rain', 'occlusion_sun']
    # go one platform at a time
    # how many possibilities?
    # how do i represent this?

    # indicator vector with length of # possible configurations on grid
    # need to discretize the number of possible configurations

    # generate platforms
    global platforms
    platforms = generate_platform.main()

    # X_constraint, y_constraint = compute_possible_positions(supports, platforms)
    #print(y_constraint.shape)
    #y_constraint = y_constraint.reshape(y_constraint.shape[0],1)
    # make GP model for constraint

    #kern = GPy.kern.RBF(input_dim=10,ARD=True)
    #constraint_model = GPy.models.GPRegression(X_constraint,y_constraint,kernel=kern)
    #mean, var = regr.predict()
    constraint_model = GPModel(optimize_restarts=5, verbose=False)
    model = GPModel(optimize_restarts=1, verbose=False)
    lower_xy = -6
    upper_xy = 6

    lower_h = 4.5
    upper_h = 19
    bounds = [{'name': 'x0', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'y0', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'x1', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'y1', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'x2', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'y2', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'x3', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'y3', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'x4', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             {'name': 'y4', 'type': 'continuous', 'domain': (lower_xy, upper_xy)},
             #{'name': 'h1', 'type': 'continuous', 'domain': (lower_h, upper_h)},
             #{'name': 'h2', 'type': 'continuous', 'domain': (lower_h, upper_h)},
             #{'name': 'h3', 'type': 'continuous', 'domain': (lower_h, upper_h)}
             ]
    design_space = GPyOpt.Design_space(space = bounds)
    acquisition_optimizer = AcquisitionOptimizer(design_space)
    acquisition = jitter_integrated_EI(model=model,constraint_model=constraint_model,space=design_space,optimizer=acquisition_optimizer)
    evaluator = Sequential(acquisition)
    global xy_ind
    xy_ind = list(chain.from_iterable((i, i + 1) for i in range(0, 50, 10)))
    global h_ind
    h_ind = [50, 52, 51]
    global all_inds
    all_inds = xy_ind + h_ind

    resi=run_bo_(platforms,x_train_red,x_true, bottom_top_heights,output_attrib,design_space,
                 model=model,acquisition=acquisition,
                 evaluator=evaluator,
                 tol=0.009,supports=supports)
    platforms[0,xy_ind] = resi.reshape(len(xy_ind),)
    toptobottom.plot_platform_grids(platforms, x_train_red, v=5, u=8,
                        list_att=['OccRain', 'OccSun', 'Surf', 'Outline'], list_plot_x=[[0, 1], [2, 3]],
                        samp_to_plot=0, grids=supports)

    



if __name__ == "__main__":
    main()
