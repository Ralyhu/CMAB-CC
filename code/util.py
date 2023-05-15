from uncertain_cc_instance import *
import correlation_clustering
from igraph import *
import numpy as np
from pyccalg import _vertex_pair_id, _vertex_pair_ids, _solve_lp_scipy, round_charikar, _lp_solution_cost

def avg_pivot_run(u_cc_instance, n_runs):
    w_plus = u_cc_instance.get_w_plus()
    w_minus = u_cc_instance.get_w_minus()
    sum_losses = 0
    clusterings = [None] * n_runs
    losses = [None] * n_runs
    for i in range(n_runs):
        clustering = correlation_clustering.min_cc_ailon(u_cc_instance, w_plus, w_minus)
        cur_loss = u_cc_instance.analytically_expected_loss(clustering, w_plus, w_minus)
        sum_losses += cur_loss
        # save the loss and the clustering for retrieving the closest to the mean
        clusterings[i] = clustering
        losses[i] = cur_loss
    avg_loss = sum_losses / n_runs
    best_index = -1
    for i in range(n_runs):
        if best_index != -1 or abs(losses[i] - avg_loss) < abs(losses[best_index] - avg_loss):
            best_index = i
    return avg_loss, losses[best_index], clusterings[best_index]

def build_A(num_vertices):
        vertex_pairs = int(num_vertices*(num_vertices-1)/2)
        A = []
        for i in range(num_vertices-1):
            for j in range(i+1,num_vertices):
                ij = _vertex_pair_id(i,j,num_vertices)
                for k in range(num_vertices):
                    if k != i and k != j:
                        ik = _vertex_pair_id(i,k,num_vertices)
                        kj = _vertex_pair_id(k,j,num_vertices)
                        #for all vertex pairs {i,j} and all vertices k \notin {i,j}, state the following triangle-inequality constraint:
                        # xij <= xik + xkj <=> xij - xik - xkj = 0
                        a = [0]*vertex_pairs
                        a[ij] = 1
                        a[ik] = -1
                        a[kj] = -1
                        A.append(a)
        return A

def _graph_to_dict(u_cc_instance, estimate_p_plus, estimate_p_minus):
    estimate_p_plus = estimate_p_plus.tolist()
    estimate_p_minus = estimate_p_minus.tolist()

    tot_min = 0
    id2vertex = {}
    vertex2id = {}
    edges = []
    graph = {}
    vertex_id = 0
    graph_igraph = u_cc_instance.get_graph()
    # TODO: optimization, most of the following indeces can be built at the beginning
    for edge in graph_igraph.es:
        u = edge.source
        v = edge.target
        eid = edge.index
        wp = estimate_p_plus[eid]
        wn = estimate_p_minus[eid]
        if u not in vertex2id:
            vertex2id[u] = vertex_id
            id2vertex[vertex_id] = u
            vertex_id += 1
        if v not in vertex2id:
            vertex2id[v] = vertex_id
            id2vertex[vertex_id] = v
            vertex_id += 1
        uid = vertex2id[u]
        vid = vertex2id[v]
        if uid < vid:
            edges.append((uid,vid))
        else:
            edges.append((vid,uid))
        if uid not in graph.keys():
            graph[uid] = {}
        if vid not in graph.keys():
            graph[vid] = {}
        min_pn = min(wp,wn)
        tot_min += min_pn
        graph[uid][vid] = (wp-min_pn,wn-min_pn)
        graph[vid][uid] = (wp-min_pn,wn-min_pn)
    # for k,v in graph.items():
    #     print(k, v)
    # print("----")
    # for edge in graph_igraph.es:
    #     print(vertex2id[edge.source], vertex2id[edge.target], estimate_p_plus[edge.index], estimate_p_minus[edge.index])
    return (id2vertex,vertex2id,edges,graph,tot_min)

def best_charikar_run(u_cc_instance, estimate_p_plus, estimate_p_minus, A, n_runs):
    #tot_min = np.sum(np.minimum(estimate_p_plus, estimate_p_minus))
    #graph_igraph = u_cc_instance.get_graph()
    num_vertices = u_cc_instance.get_number_nodes()
    vertex_pairs = int(num_vertices*(num_vertices-1)/2)

    (id2vertex,vertex2id,edges,graph,tot_min) = _graph_to_dict(u_cc_instance, estimate_p_plus, estimate_p_minus)

    # build linear program then call methods.., A was already built
    b = [0]*len(A)
    c = [0]*vertex_pairs
    
    for (u,v) in edges:
        uv = _vertex_pair_id(u,v,num_vertices)
        (wp,wn) = graph[u][v]
        if wp != wn:
            if wp < wn: #(u,v) \in E^-
                c[uv] = -(wn-wp)
            else: #(u,v) \in E^+
                c[uv] = (wp-wn)
    # solve linear program
    (lp_var_assignment,obj_value) = _solve_lp_scipy(A,b,c)

    lp_cost = _lp_solution_cost(lp_var_assignment,graph,num_vertices) + tot_min

    id2vertexpair = _vertex_pair_ids(num_vertices)
    # print(id2vertex)
    # id2vertex = {}
    # for i in range(num_vertices):
    #     id2vertex[i] = i
    # print(id2vertex)
    # print("-----------")

    best_clustering = None
    best_loss = float("inf")
    for _ in range(n_runs):
        clusters = round_charikar(lp_var_assignment,id2vertexpair,id2vertex, edges, graph,lp_cost-tot_min) # 'edges' parameter can be None since it is not used in the method
        #print(clusters)
        # convert clusters into cluster membership array and map nodes id to original ids...
        clustering = [None] * num_vertices
        id_cluster = 0
        for cluster in clusters:
            for node in cluster:
                clustering[id2vertex[node]] = id_cluster
            id_cluster += 1
        #print(clustering)
        cur_loss = u_cc_instance.analytically_expected_loss(clustering, estimate_p_plus, estimate_p_minus)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_clustering = clustering
    return best_clustering

def best_pivot_run(u_cc_instance, estimate_p_plus, estimate_p_minus, n_runs):
    best_clustering = None
    best_loss = float("inf")
    for _ in range(n_runs):
        clustering = correlation_clustering.min_cc_ailon(u_cc_instance, estimate_p_plus, estimate_p_minus)
        cur_loss = u_cc_instance.analytically_expected_loss(clustering, estimate_p_plus, estimate_p_minus)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_clustering = clustering
    return best_clustering

def check_mean_estimates(u_cc_instance, estimates_plus, estimates_minus, n_plus, n_minus):
    graph = u_cc_instance.get_graph()
    estimates_plus = estimates_plus.tolist()
    estimates_minus = estimates_minus.tolist()
    n_plus = n_plus.tolist()
    n_minus = n_minus.tolist()
    for edge in graph.es:
        eid = edge.index
        print("(% 3d, % 3d) // + % .2f/% .2f (% 4d) - % .2f/% .2f (% 4d)" %(edge.source, edge.target, edge["p+"], estimates_plus[eid], n_plus[eid], edge["p-"], estimates_minus[eid], n_minus[eid]))
    
def count_clusters(clustering):
    clusters = set()
    for c in clustering:
        clusters.add(c)
    return len(clusters)

def count_majority_positive(u_cc_instance):
    graph = u_cc_instance.get_graph()
    n_edges = u_cc_instance.get_number_edges()
    cont = 0
    for edge in graph.es:
        p_plus = edge["p+"]
        p_minus = edge["p-"]
        if p_plus > p_minus:
            cont += 1
    return cont, n_edges

def count_internal_external_edges(graph, cluster_membership):
    n_edges = len(graph.es)
    same_cluster_mask = np.array([cluster_membership[edge.source] == cluster_membership[edge.target] for edge in graph.es])
    intra_community_edges = same_cluster_mask.sum()
    inter_community_edges = n_edges - intra_community_edges
    return intra_community_edges, inter_community_edges

def get_size_clusters(cluster_membership):
    res = {}
    for label in cluster_membership:
        try:
            res[label] += 1
        except:
            res[label] = 1
    return res

def get_clustering_info(clustering):
    d = get_size_clusters(clustering)
    n_clusters = len(d.keys())
    avg_size_clusters = sum(d.values())/n_clusters
    return n_clusters, avg_size_clusters