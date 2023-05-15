from .cmab import CMAB
from util import *
import constants
import util
import random
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

class EpsilonGreedy(CMAB):

    # by specifying the "exploration_probability" parameter, the algorithm maintains a fixed rate of exploration
    # otherwise, the exploration steps are less frequent as rounds passes
    def __init__(self, u_cc_instance, epsilon=5, exploration_probability=None):
        super().__init__(u_cc_instance)
        self.epsilon = epsilon
        self.exploration_proability=exploration_probability
        self.rand_gen = random.Random(constants.seed)

    def select_action(self, t):
        p = self.rand_gen.uniform(0,1)
        if self.exploration_proability:
            threshold = self.exploration_proability
        else:
            threshold = self.epsilon/(t+1)
        if p <= threshold:
            return self.random_cluster()
        else:
            if self.oracle == "pivot":
                return util.best_pivot_run(self.u_cc_instance, self.estimate_p_plus, self.estimate_p_minus, self.num_trials_pivot)
            return util.best_charikar_run(self.u_cc_instance, self.estimate_p_plus, self.estimate_p_minus, self.A, self.num_trials_pivot)

    def random_cluster(self):
        return self.generate_random_cluster(self.u_cc_instance)

    def generate_random_cluster(self, u_cc_instance):
        graph = u_cc_instance.get_graph()
        num_nodes = u_cc_instance.get_number_nodes()
        cluster_membership = [-1] * num_nodes
        graph = u_cc_instance.get_graph()
        # generate random permutation of nodes (Fisher-Yates algorithm O(n))
        permutation = np.arange(num_nodes)
        np.random.shuffle(permutation)
        permutation = permutation.tolist()
        index = 0
        cluster_id = 0
        while index < num_nodes:
            cur_node = permutation[index]
            # get neighbors cluster id + the next free one and sample randomly the cluster for cur_node
            neighbors = graph.neighbors(cur_node)
            neighs_cluster_ids = [cluster_membership[neigh] for neigh in neighbors if cluster_membership[neigh] != -1]
            neighs_cluster_ids.append(cluster_id)
            cur_cluster = self.rand_gen.choice(neighs_cluster_ids)
            cluster_membership[cur_node] = cur_cluster
            if cur_cluster == cluster_id:
                cluster_id += 1
            # get next valid index
            while index < num_nodes and cluster_membership[permutation[index]] != -1:
                index += 1
        # check conditions
        #assert all(x != -1 for x in cluster_membership) == True
        #assert index == num_nodes
        return cluster_membership