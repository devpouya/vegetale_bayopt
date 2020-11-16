import numpy as np
import os
import time
import pickle as pk

import torch

from ..test_functions.experiment_configuration import sample_init_points, \
    generate_ising_interaction
from ..test_functions.binary_categorical import spin_covariance, partition, ising_dense

import src.python.utils_vegetale as uv

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25
CENTROID_N_CHOICE = 3
CENTROID_GRID = (4, 4)
CENTROID_N_EDGES = CENTROID_GRID[0] * (CENTROID_GRID[1] - 1) + (CENTROID_GRID[0] - 1) * CENTROID_GRID[1]

list_attrib = ['occlusion_rain', 'occlusion_sun',
               'outline_lengths_0', 'outline_lengths_1', 'outline_lengths_2', 'outline_lengths_3',
               'outline_lengths_4',
               'surface_areas_0', 'surface_areas_1', 'surface_areas_2', 'surface_areas_3', 'surface_areas_4']

output_attrib = ['occlusion_rain', 'occlusion_sun']


def edge_choice(x, interaction_list):
    edge_weight = np.zeros(x.shape)
    for i in range(len(interaction_list)):
        edge_weight[x == i] = np.hstack([interaction_list[i][0].reshape(-1), interaction_list[i][1].reshape(-1)])[
            x == i]
    grid_h, grid_w = CENTROID_GRID
    split_ind = grid_h * (grid_w - 1)
    return edge_weight[:split_ind].reshape((grid_h, grid_w - 1)), edge_weight[split_ind:].reshape((grid_h - 1, grid_w))


class Centroid(object):
    """
    Ising Sparsification Problem with the simplest graph
    """

    def __init__(self, random_seed_pair=(None, None)):
        self.n_vertices = np.array([CENTROID_N_CHOICE] * CENTROID_N_EDGES)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init,
                                         sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0),
                                                            random_seed_pair[1]).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join(
            [str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        self.interaction_list = []
        self.covariance_list = []
        self.partition_original_list = []
        self.n_ising_models = 3
        ising_seeds = np.random.RandomState(random_seed_pair[0]).randint(0, 10000, (self.n_ising_models,))
        for i in range(self.n_ising_models):
            interaction = generate_ising_interaction(CENTROID_GRID[0], CENTROID_GRID[1], ising_seeds[i])
            interaction = (interaction[0].numpy(), interaction[1].numpy())
            covariance, partition_original = spin_covariance(interaction, CENTROID_GRID)
            self.interaction_list.append(interaction)
            self.covariance_list.append(covariance)
            self.partition_original_list.append(partition_original)

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        interaction_mixed = edge_choice(x.numpy(), self.interaction_list)
        partition_mixed = partition(interaction_mixed, CENTROID_GRID)
        kld_sum = 0
        for i in range(self.n_ising_models):
            kld = ising_dense(interaction_sparsified=interaction_mixed, interaction_original=self.interaction_list[i],
                              covariance=self.covariance_list[i], partition_sparsified=partition_mixed,
                              partition_original=self.partition_original_list[i], grid_h=CENTROID_GRID[0])
            kld_sum += kld
        return float(kld_sum / float(self.n_ising_models)) * x.new_ones((1,)).float()


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0
    print("X IN PEST CONTROL IS  {}".format(x))
    init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(object):
    """
    Ising Sparsification Problem with the simplest graph
    """

    def __init__(self, random_seed=None):
        self.n_vertices = np.array([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init,
                                         sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0),
                                                            random_seed).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4)
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy())
        return float(evaluation) * x.new_ones((1,)).float()


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


def _vegetale_score(x, x_,bottom_top_heights, rain_occ, sun_occ, normalizer):
    vals = GH_attr(x, x_,bottom_top_heights)
    xgh = vals[output_attrib]
    xgh = np.asarray(xgh.values)
    print("found values {}".format(xgh))
    xgh = normalizer.transform(xgh)
    occs = normalizer.transform(np.array([rain_occ, sun_occ]).reshape(1, 2))

    diff_1 = occs[0, 0] - xgh[0, 0]
    diff_2 = occs[0, 1] - xgh[0, 1]

    loss = np.sqrt(diff_2 ** 2 + diff_1 ** 2) / 2
    # ITER += 1
    #print(loss)
    return loss


class Vegetale(object):
    """
    Vegetale
    """

    def __init__(self, space, platforms, x_,bottom_top_heights, rain_occ, sun_occ, random_seed=None, normalizer=None):
        self.n_vertices = np.array([len(s) for s in space])
        #print(self.n_vertices)
        self.x_ = x_
        self.suggested_init = torch.empty(0).long()

        self.suggested_init = torch.cat([self.suggested_init,
                                         sample_init_points(self.n_vertices, 30 - self.suggested_init.size(0),
                                                            random_seed).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.normalizer = normalizer
        self.random_seed_info = str(random_seed).zfill(4)
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            # adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            adjmat = torch.ones(n_v, n_v) - torch.diag(torch.ones(n_v))
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)

        # legal positions for each platform
        self.design_space1 = space[0]
        self.design_space2 = space[1]
        self.design_space3 = space[2]
        self.design_space4 = space[3]
        self.design_space5 = space[4]

        self.radii1 = platforms[0, platforms[0, :] > 0.0]  # np.where(platforms[0, :] > 0.0)[0]
        self.radii2 = platforms[1, platforms[1, :] > 0.0]  # np.where(platforms[1, :] > 0.0)[0]
        self.radii3 = platforms[2, platforms[2, :] > 0.0]  # np.where(platforms[2, :] > 0.0)[0]
        self.radii4 = platforms[3, platforms[3, :] > 0.0]  # np.where(platforms[3, :] > 0.0)[0]
        self.radii5 = platforms[4, platforms[4, :] > 0.0]  # np.where(platforms[4, :] > 0.0)[0]

        self.bottom_top_heights = bottom_top_heights

        self.rain_occ = rain_occ
        self.sun_occ = sun_occ

    def evaluate(self, x):
        # assert x.numel() == len(self.n_vertices)
        # if x.dim() == 2:
        #    x = x.squeeze(0)
        #perm1 = np.argwhere(np.array(self.design_space1[x[0]]) == 1)
        ## print("p1 {}".format(perm1))
        ## print("jj {}".format(self.design_space1[x[0]]))
        #perm2 = np.argwhere(np.array(self.design_space2[x[1]]) == 1)
        #perm3 = np.argwhere(np.array(self.design_space3[x[2]]) == 1)
        #perm4 = np.argwhere(np.array(self.design_space4[x[3]]) == 1)
        #perm5 = np.argwhere(np.array(self.design_space5[x[4]]) == 1)
        ## print([perm1,perm2,perm3,perm4,perm5])
        #platforms_eval = np.zeros((5, 11))
        #for i, ind in enumerate(perm1):
        #    platforms_eval[0, ind] = self.radii1[i]
        #for i, ind in enumerate(perm2):
        #    platforms_eval[1, ind] = self.radii2[i]
        #for i, ind in enumerate(perm3):
        #    platforms_eval[2, ind] = self.radii3[i]
        #for i, ind in enumerate(perm4):
        #    platforms_eval[3, ind] = self.radii4[i]
        #for i, ind in enumerate(perm5):
        #    platforms_eval[4, ind] = self.radii5[i]


        platforms = np.zeros((1, 58))
        platforms[:,:11] = self.design_space1[x[0]]
        platforms[:,11:22] = self.design_space2[x[1]]
        platforms[:,22:33] = self.design_space3[x[2]]
        platforms[:,33:44] = self.design_space4[x[3]]
        platforms[:,44:55] = self.design_space5[x[4]]
        #platforms[:, :-3] = platforms_eval.reshape(1, 55)
        platforms[:, -1] = 15
        platforms[:, -2] = 11.5
        platforms[:, -3] = 7.5
        #print(platforms)
        loss = _vegetale_score(platforms, self.x_,self.bottom_top_heights,
                               self.rain_occ, self.sun_occ, self.normalizer)

        return float(loss) * x.new_ones((1,)).float()


if __name__ == '__main__':
    pass
    # evaluator = PestControl(5355)
    # # x = np.random.RandomState(123).randint(0, 5, (PESTCONTROL_N_STAGES, ))
    # # print(_pest_control_score(x))
    # n_evals = 2000
    # for _ in range(10):
    # 	best_pest_control_loss = float('inf')
    # 	for i in range(n_evals):
    # 		if i < evaluator.suggested_init.size(0):
    # 			random_x = evaluator.suggested_init[i]
    # 		else:
    # 			random_x = torch.Tensor([np.random.randint(0, 5) for h in range(len(evaluator.n_vertices))]).long()
    # 		pest_control_loss = evaluator.evaluate(random_x).item()
    # 		if pest_control_loss < best_pest_control_loss:
    # 			best_pest_control_loss = pest_control_loss
    # 	print('With %d random search, the pest control objective(%d stages) is %f' % (n_evals, PESTCONTROL_N_STAGES, best_pest_control_loss))

    # for _ in range(10):
    # 	x = torch.from_numpy(np.random.RandomState(None).randint(0, 3, (ALTERATION_N_EDGES, )))
    # 	print(evaluator.evaluate(x))
    n_evals = 100
    for _ in range(10):
        evaluator = Centroid((9154, None))
        min_eval = float('inf')
        for i in range(n_evals):
            if i < 2:  # evaluator.suggested_init.size(0):
                random_x = evaluator.suggested_init[i]
            else:
                random_x = torch.Tensor([np.random.randint(0, 5) for h in range(len(evaluator.n_vertices))]).long()
            evaluation = evaluator.evaluate(random_x).item()
            if evaluation < min_eval:
                min_eval = evaluation
        print('With %d random search, the ising alteration objective(%d edges) is %f' % (
            n_evals, CENTROID_N_EDGES, min_eval))
