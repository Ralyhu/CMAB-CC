from .cmab import CMAB
import numpy as np
from numpy.random import beta
from numpy.random import binomial
import util

class CTS(CMAB):

    def __init__(self, u_cc_instance):
        super().__init__(u_cc_instance)
    
    def init_estimates(self):
        n_edges = self.u_cc_instance.get_number_edges()
        self.a_plus = np.ones(n_edges)
        self.b_plus = np.ones(n_edges)
        self.a_minus = np.ones(n_edges)
        self.b_minus = np.ones(n_edges)

        self.n_plus = np.zeros(shape=n_edges)
        self.n_minus = np.zeros(shape=n_edges)

    def select_action(self, t):
        theta_plus = beta(self.a_plus, self.b_plus)
        theta_minus = beta(self.a_minus, self.b_minus)
        #assert (theta_plus >= 0).all() and (theta_plus <= 1).all() and (theta_minus >= 0).all() and (theta_minus <= 1).all() == True
        if self.oracle == "pivot":
            return util.best_pivot_run(self.u_cc_instance, theta_plus, theta_minus, self.num_trials_pivot)
        return util.best_charikar_run(self.u_cc_instance, theta_plus, theta_minus, self.A, self.num_trials_pivot)

    def update_estimates(self, clustering, t):
        igraph_graph = self.u_cc_instance.get_graph()
        optimal_same_cluster_mask = self.optimal_same_cluster_mask
        optimal_different_cluster_mask = self.optimal_different_cluster_mask
        n_edges = self.u_cc_instance.get_number_edges()
        M = self.M

        same_cluster_mask = np.array([clustering[edge.source] == clustering[edge.target] for edge in igraph_graph.es])
        different_cluster_mask = (np.ones(shape=n_edges) - same_cluster_mask)

        # variables updated for debug only, not used by CTS
        self.n_plus = self.n_plus + different_cluster_mask
        self.n_minus = self.n_minus + same_cluster_mask

        ones = np.ones(n_edges)
        filtered_sample_plus = self.sample_plus * different_cluster_mask
        filtered_sample_minus = self.sample_minus * same_cluster_mask
        # normalize by M
        filtered_sample_plus = filtered_sample_plus / M
        filtered_sample_minus = filtered_sample_minus / M
        # sample a set of bernoulli random variables
        y_plus = binomial(n=1, p=filtered_sample_plus)
        y_minus = binomial(n=1, p=filtered_sample_minus)

        self.a_plus = self.a_plus + different_cluster_mask * y_plus
        self.b_plus = self.b_plus + different_cluster_mask * (ones - y_plus)
        self.a_minus = self.a_minus + same_cluster_mask * y_minus
        self.b_minus = self.b_minus + same_cluster_mask * (ones - y_minus)

        cumulative_interaction = np.sum(same_cluster_mask * self.sample_minus + different_cluster_mask * self.sample_plus)
        cumulative_interaction_optimal = np.sum(optimal_same_cluster_mask * self.sample_minus + optimal_different_cluster_mask * self.sample_plus)
        cumulative_interaction_expected = np.sum(same_cluster_mask * self.u_cc_instance.get_w_minus() + different_cluster_mask * self.u_cc_instance.get_w_plus())

        n_clusters, avg_size_clusters = util.get_clustering_info(clustering)
        self.n_clusters[t] = n_clusters
        self.avg_size_clusters[t] = avg_size_clusters

        self.rewards[t] = cumulative_interaction
        self.rewards_optimal[t] = cumulative_interaction_optimal
        self.rewards_expected[t] = cumulative_interaction_expected
        if t > round(len(self.rewards) * 0.9) and cumulative_interaction_expected < self.best_loss:
            self.best_loss = cumulative_interaction_expected
            self.best_clustering = clustering
        self.estimate_p_plus = self.get_estimates_plus()
        self.estimate_p_minus = self.get_estimates_minus()
        self.compute_error(t)

    # builds mean estimates with a and b vectors
    def get_estimates_plus(self):
        return self.a_plus / (self.a_plus + self.b_plus)
        
    def get_estimates_minus(self):
        return self.a_minus / (self.a_minus + self.b_minus)