import numpy as np
import torch

class vegetale(object):
    """
        Objective function based on the MSE Loss
        of the sun and rain occlusion
    """

    def __init__(self, random_seed=None):
        # number of total vertices
        # = product of all list lengths
        self.n_vertices = np.array([0])
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []

    def evaluate(self, x):
        # map the indicator to geometry
        # call GH and get values
        # compare to real values
        pass

