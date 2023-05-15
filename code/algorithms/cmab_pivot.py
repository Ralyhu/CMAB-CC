from os import system
from .cmab import CMAB
import numpy as np
import util

class PCEXPCLCB(CMAB):

    def __init__(self, interaction_graph):
        super().__init__(interaction_graph)

    def init_estimates(self):
        self.igraph_graph = self.u_cc_instance.get_graph()
        n_edges = self.u_cc_instance.get_number_edges()
        self.n_plus = np.zeros(shape=n_edges)
        self.estimate_p_plus = np.zeros(shape=n_edges)
        #self.estimate_p_plus = np.ones(shape=n_edges)
        self.n_minus = np.zeros(shape=n_edges)
        #self.estimate_p_minus = np.ones(shape=n_edges)
        self.estimate_p_minus = np.zeros(shape=n_edges)
        self.sample_plus = np.zeros(shape=n_edges)
        self.sample_minus = np.zeros(shape=n_edges)

    def select_action(self, t):
        counters_sum = self.n_plus + self.n_minus
        gamma_plus = np.divide(self.n_plus, counters_sum, out=np.full_like(counters_sum, 0.5, dtype=np.double), where=counters_sum!=0)
        gamma_minus = 1 - gamma_plus
        delta = 1 - self.estimate_p_plus - self.estimate_p_minus

        corrected_estimates_plus = np.where(delta > 0, self.estimate_p_plus + delta * gamma_plus, self.estimate_p_plus + delta * gamma_minus)
        corrected_estimates_minus = np.where(delta > 0, self.estimate_p_minus + delta * gamma_minus, self.estimate_p_minus + delta * gamma_plus)
        #assert np.allclose(corrected_estimates_plus + corrected_estimates_minus, np.ones_like(corrected_estimates_minus), rtol=1e-05, atol=1e-08, equal_nan=False) == True
        return util.best_pivot_run(self.u_cc_instance, corrected_estimates_plus, corrected_estimates_minus, self.num_trials_pivot)
