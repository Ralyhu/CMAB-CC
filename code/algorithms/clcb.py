from .cmab import CMAB
import numpy as np
import math
import util

class CLCB(CMAB):

    def __init__(self, interaction_graph, m_variant=False):
        super().__init__(interaction_graph)
        if m_variant:
            self.uncertain_constant = 0.5
        else:
            self.uncertain_constant = 1.5

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
        n_edges = self.u_cc_instance.get_number_edges()
        const = np.full(shape=n_edges, fill_value=self.uncertain_constant * math.log(t+1))
        uncertainty_plus = np.sqrt(np.divide(const, self.n_plus, out=np.zeros_like(const), where=self.n_plus!=0))
        uncertainty_minus = np.sqrt(np.divide(const, self.n_minus, out=np.zeros_like(const), where=self.n_minus!=0))
        corrected_estimates_plus = self.estimate_p_plus - uncertainty_plus
        corrected_estimates_minus = self.estimate_p_minus - uncertainty_minus
        zeros = np.zeros(shape=n_edges)
        if self.oracle == "pivot":
            return util.best_pivot_run(self.u_cc_instance, np.maximum(zeros, corrected_estimates_plus), np.maximum(zeros, corrected_estimates_minus), self.num_trials_pivot)
        return util.best_charikar_run(self.u_cc_instance, np.maximum(zeros, corrected_estimates_plus), np.maximum(zeros, corrected_estimates_minus), self.A, self.num_trials_pivot)