from uncertain_cc_instance import *
from collections import *
import constants
import random
import numpy as np
import math
import scipy
import time

# Hill climbing step
def optimize_clustering_by_relocations(u_cc_instance, estimate_w_plus, estimate_w_minus, clustering):
    # NB: assume the estimates are given as lists
    num_nodes = u_cc_instance.get_number_nodes()
    max_iters = constants.maximum_number_iterations_relocation
    graph = u_cc_instance.get_graph()
    # ref for avoiding dots
    incident = graph.incident
    edges = graph.es

    has_changed = True
    cur_it = 0
    next_id_cluster = max(clustering) + 1
    while cur_it < max_iters and has_changed:
        copy_starting = list(clustering) 
        # random order for nodes evaluation
        permutation = np.arange(num_nodes)
        np.random.shuffle(permutation)
        permutation = permutation.tolist()
        
        has_changed = False
        for cur_node in permutation:
            gain_by_cluster = defaultdict(int)
            c_u = clustering[cur_node]
            # get neighbors of current node
            incident_edges = incident(cur_node)
            for edge_id in incident_edges:
                edge = edges[edge_id]
                if edge.source != cur_node:
                    neigh = edge.source
                else: 
                    neigh = edge.target
                puv_plus = estimate_w_plus[edge_id]
                puv_minus = estimate_w_minus[edge_id]
                c_neigh = clustering[neigh]
                # delta plus
                delta = puv_plus - puv_minus
                if c_u == c_neigh:
                    # need delta minus, change sign to delta
                    delta = - delta 
                gain_by_cluster[c_neigh] += delta
            # intra cluster gain by moving to a new cluster
            gain_by_cluster[next_id_cluster] = 0
            
            # compute delta f(cur_node) by taking the max over neighbor clusters
            # first remove the entry associated to current cluster (if cur_node is linked to another node in c_u)
            if c_u in gain_by_cluster:
                delta_cur = gain_by_cluster.pop(c_u)
            else:
                delta_cur = 0
            assert len(gain_by_cluster.keys()) != 0
            cur_max_value = float("-inf")
            cur_max_cluster = -1
            for neigh_cluster, value in gain_by_cluster.items():
                assert neigh_cluster != c_u
                if value > cur_max_value:
                    cur_max_value = value
                    cur_max_cluster = neigh_cluster
            assert cur_max_cluster != -1
            delta_f = delta_cur + cur_max_value
            if delta_f > 0:
                # move cur_node from current cluster to the best one
                has_changed = True
                clustering[cur_node] = cur_max_cluster
                # next id for moving to new cluster
                if cur_max_cluster == next_id_cluster:
                    next_id_cluster += 1
                    #print("Node " + str(cur_node) + " moved to a new cluster with id " + str(cur_max_cluster))
        cur_it += 1
        #assert u_cc_instance.analytically_expected_loss(clustering, u_cc_instance.get_p_plus(), u_cc_instance.get_p_minus()) <= u_cc_instance.analytically_expected_loss(copy_starting, u_cc_instance.get_p_plus(), u_cc_instance.get_p_minus())
    assert cur_it <= max_iters
    #print(cur_it)
    return clustering, cur_it


# MIN-CC Pivot algorithm
def min_cc_ailon(u_cc_instance, estimate_w_plus, estimate_w_minus, final_optimization = False, file_path_times=None):
    """Computes a clustering using the min-cc algorithm of Ailon et al. (2015)
    
    Arguments:
        u_cc_instance {[type]} -- [description]                                                                  
    """
    start1 = time.time()
    estimate_w_plus = estimate_w_plus.tolist()
    estimate_w_minus = estimate_w_minus.tolist()
    M = u_cc_instance.get_M()
    num_nodes = u_cc_instance.get_number_nodes()
    permutation = np.arange(num_nodes)
    #permutation = list(range(0, num_nodes))
    marked_nodes = [False] * num_nodes
    cluster_membership = [-1] * num_nodes
    graph = u_cc_instance.get_graph()
    # generate random permutation of nodes (Fisher-Yates algorithm O(n))
    #constants.rand_gen.shuffle(permutation)
    #np.random.seed()
    np.random.shuffle(permutation)
    permutation = permutation.tolist()
    cluster_id = 0
    # ref for avoiding dots
    incident = graph.incident
    edges = graph.es
    for cur_node in permutation:
        if not marked_nodes[cur_node]:
            cluster_membership[cur_node] = cluster_id
            marked_nodes[cur_node] = True
            # get neighbors of current node
            incident_edges = incident(cur_node)
            for edge_id in incident_edges:
                edge = edges[edge_id]
                if edge.source != cur_node:
                    neigh = edge.source
                else: 
                    neigh = edge.target
                #assert neigh != cur_node
                if not marked_nodes[neigh]:
                    wuv_plus = estimate_w_plus[edge.index]
                    wuv_minus = estimate_w_minus[edge.index]
                    #assert math.isclose(w_plus + w_minus, 1.0, rel_tol=1e-6)
                    if wuv_plus > wuv_minus:
                        cluster_membership[neigh] = cluster_id
                        marked_nodes[neigh] = True
            cluster_id += 1
    total_time1 = time.time() - start1
    # check conditions
    #assert all(x != -1 for x in cluster_membership) == True
    #assert all(x == True for x in marked_nodes)
    if final_optimization:
        start2 = time.time()
        _, n_it = optimize_clustering_by_relocations(u_cc_instance, estimate_w_plus, estimate_w_minus, cluster_membership)
        total_time2 = time.time() - start2
        if file_path_times != None:
            file2 = file_path_times + "times2.txt"
            with open(file2, "w+") as f:
                f.write(str(total_time2) + "\n")
                f.write(str(n_it))
    if file_path_times != None:
        file1 = file_path_times + "times1.txt"
        with open(file1, "w+") as f:
            f.write(str(total_time1))
    return cluster_membership
