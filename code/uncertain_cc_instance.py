from igraph import *
from scipy.stats import truncnorm 
from scipy.stats import *
import numpy as np
from numpy.random import binomial as binomial

# pylint: disable=E1133
# pylint: disable=E1137
class UncertainCCInstance:
    
    """main constructor which builds an UncertainCCInstance from an igraph graph object and maximum strength of interaction M
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, graph, M = 1, edge_distributions = "bernoulli"):
        self.M = M
        self.distributions_type = edge_distributions
        self.graph = graph
        degrees = graph.degree()
        # plus one because then i multiply by some probability in order to sample by degree
        self.degrees = [d+1 for d in degrees]
        degrees_positives = self.get_degrees_positives()
        self.degrees_positives = [d+1 for d in degrees_positives]
        degrees_negatives = self.get_degrees_negatives()
        self.degrees_negatives = [d+1 for d in degrees_negatives]
        self.w_plus = np.array(graph.es["w+_expected"])
        self.w_minus = np.array(graph.es["w-_expected"])
        if edge_distributions == "gaussian":
            self.std_plus = np.array(graph.es["w+_var"])
            self.std_minus = np.array(graph.es["w-_var"])
            self.a_plus, self.b_plus = (0 - self.w_plus) / self.std_plus, (M - self.w_plus) / self.std_plus
            self.a_minus, self.b_minus = (0 - self.w_minus) / self.std_minus, (M - self.w_minus) / self.std_minus

    @staticmethod
    def get_instance_from_file(path):
        with open(path) as f:
            header = f.readline()
            header_splitted = header.split(" ")
            n_nodes = int(header_splitted[0])
            n_edges = int(header_splitted[1])
            g = Graph()
            g.add_vertices(n_nodes)
            edges = [None] * n_edges
            w_plus_means = [None] * n_edges
            w_minus_means = [None] * n_edges
            id_edge = 0
            for line in f.readlines():
                line_splitted = line.split(" ")
                i = int(line_splitted[0])
                j = int(line_splitted[1])
                w_plus = float(line_splitted[2])
                w_minus = float(line_splitted[3])
                edges[id_edge] = (i, j)
                w_plus_means[id_edge] = w_plus
                w_minus_means[id_edge] = w_minus
                id_edge += 1
            g.add_edges(edges)
            g.es["w+_expected"] = w_plus_means
            g.es["w-_expected"] = w_minus_means
            # TODO: assume distribuz. bernoulli sugli archi per ora
            return UncertainCCInstance(g)

    def sample_graph_all(self):
        if self.distributions_type == "bernoulli":
            sample_plus = binomial(1, self.graph.es["w+_expected"])
            sample_minus = binomial(1, self.graph.es["w-_expected"]) 
        elif self.distributions_type == "gaussian":
            sample_plus = truncnorm.rvs(a=self.a_plus, b=self.b_plus, loc=self.graph.es["w+_expected"], scale=self.graph.es["w+_var"])
            sample_minus = truncnorm.rvs(a=self.a_minus, b=self.b_minus, loc=self.graph.es["w-_expected"], scale=self.graph.es["w-_var"])
        return sample_plus, sample_minus

    def analytically_expected_loss(self, cluster_membership, w_plus, w_minus):
        # compute expected cumulative loss analytically
        graph = self.graph
        n_edges = self.get_number_edges()
        
        same_cluster_mask = np.array([cluster_membership[edge.source] == cluster_membership[edge.target] for edge in graph.es])
        different_cluster_mask = (np.ones(shape=n_edges) - same_cluster_mask)
        objective = same_cluster_mask * w_minus + different_cluster_mask * w_plus
        loss = np.sum(objective)
        # test con modifica assunzione sui non-linked pairs
        # size_clusters = util.get_size_clusters(cluster_membership)
        # int_edges = {}
        # for edge in graph.es:
        #     c_u = cluster_membership[edge.source]
        #     c_v = cluster_membership[edge.target]
        #     if c_u == c_v:
        #         try:
        #             int_edges[c_u] += 1
        #         except:
        #             int_edges[c_u] = 1
        # for c, size in size_clusters.items():
        #     try:
        #         n_int_edges = int_edges[c]
        #     except:
        #         n_int_edges = 0
        #     loss += (size * (size - 1))/2 - n_int_edges
        return loss   
    
    def get_graph(self):
        return self.graph

    def get_number_nodes(self):
        return len(self.graph.vs)

    def get_degrees(self):
        # return a copy of degrees in case one modifies it
        return list(self.degrees)

    def get_degrees_positives(self):
        degrees = [0] * self.get_number_nodes()
        for edge in self.graph.es:
            if edge["w+_expected"] > edge["w-_expected"]:
                degrees[edge.source] += 1
                degrees[edge.target] += 1
        return degrees

    def get_degrees_negatives(self):
        degrees = [0] * self.get_number_nodes()
        for edge in self.graph.es:
            if edge["w-_expected"] > edge["w+_expected"]:
                degrees[edge.source] += 1
                degrees[edge.target] += 1
        return degrees


    def get_number_edges(self):
        return len(self.graph.es)

    def get_M(self):
        return self.M

    def get_w_plus(self):
        return self.w_plus
    
    def get_w_minus(self):
        return self.w_minus

    def print(self):
        #summary(self.graph)
        print(self.graph)

    def get_distributions_type(self):
        return self.distributions_type

    def get_approximate_condition(self):
        sum_weights = 0.0
        sum_weights += np.sum(self.w_plus)
        sum_weights += np.sum(self.w_minus)
        #condition_holds = sum_weights <= self.M * self.get_number_edges()
        return sum_weights
    
    def get_minimum_loss_possible(self):
        return self.get_number_edges() - np.sum(np.maximum(self.w_minus, self.w_plus))
        #return self.get_number_edges() - np.sum(np.absolute(self.w_minus-self.w_plus))
    
    
