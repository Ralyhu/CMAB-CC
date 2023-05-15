import argparse as args
import os
import time
import traceback
import sys
import networkx as nx
import math
import numpy as np
import constants
from util import *
import util
from uncertain_cc_instance import UncertainCCInstance

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
basePath = os.path.dirname(os.path.abspath(__file__)) + "/../" 

# basePath = os.getcwd() + "/data/"
# basePath_out = os.getcwd() + "/output/"

basePath_out = basePath + "/output/"
basePath_data = basePath + "/data/"

datasets = {
    "adjnoun": "adjnoun/processed",
    "amazon": "amazon/processed",
    "c-usa": "c-usa/processed",
    "dblp": "dblp/processed",
    "dolphins": "dolphins/processed",
    "epinions": "epinions/processed",
    "football": "football/processed",
    "high_school": "high_school/processed",
    "ht": "ht/processed",
    "jazz": "jazz/processed",
    "karate": "karate/processed",
    "lastfm": "lastfm/processed",
    "polbooks": "polbooks/processed",
    "primary_school": "primary_school/processed",
    "prosper-loans": "prosper-loans/processed",
    "stack-overflow": "stack-overflow/processed",
    "wiki-talk": "wiki-talk/processed",
    "wikipedia": "wikipedia/processed",
    "zebra": "zebra/processed",
    "adjnoun_PC": "adjnoun/processed_PC",
    "amazon_PC": "amazon/processed_PC",
    "c-usa_PC": "c-usa/processed_PC",
    "dblp_PC": "dblp/processed_PC",
    "dolphins_PC": "dolphins/processed_PC",
    "epinions_PC": "epinions/processed_PC",
    "football_PC": "football/processed_PC",
    "high_school_PC": "high_school/processed_PC",
    "ht_PC": "ht/processed_PC",
    "jazz_PC": "jazz/processed_PC",
    "karate_PC": "karate/processed_PC",
    "lastfm_PC": "lastfm/processed_PC",
    "polbooks_PC": "polbooks/processed_PC",
    "primary_school_PC": "primary_school/processed_PC",
    "prosper-loans_PC": "prosper-loans/processed_PC",
    "stack-overflow_PC": "stack-overflow/processed_PC",
    "wiki-talk_PC": "wiki-talk/processed_PC",
    "wikipedia_PC": "wikipedia/processed_PC",
    "zebra_PC": "zebra/processed_PC",
    "c-usa_complete": "c-usa/processed_C",
    "dolphins_complete": "dolphins/processed_C",
    "ht_complete": "ht/processed_C",
    "karate_complete": "karate/processed_C",
    "zebra_complete": "zebra/processed_C",
}

def create_parser():
    parser = args.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', help="Name of the dataset", type=str)
    parser.add_argument("-em", "--estimation_method", help="Estimation method to derive CC weights", choices=["jaccard", "adamic", "actual"], default="jaccard")
    parser.add_argument("-o", "--oracle", help="Oracle to use in CMAB framework", choices=["pivot", "charikar"], default="pivot")
    parser.add_argument("-s", "--seed", help="Seed value for random number generators", type=int, default=constants.seed)
    return parser

def adamic_adar_index(G, ebunch=None):
    r"""
    Code taken from the networkX implementation of adamic-adar
    
    Compute the Adamic-Adar index of all node pairs in ebunch.

    Adamic-Adar index of `u` and `v` is defined as

    .. math::

        \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}

    where :math:`\Gamma(u)` denotes the set of neighbors of `u`.

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Adamic-Adar index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Adamic-Adar index.

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        lcn = len(cn)
        return sum(1 / math.log(G.degree(w)) for w in cn) / lcn if lcn != 0 else 0.0

    return ((u, v, predict(u, v)) for u, v in ebunch)

def main(parsed):
    dataset_name = parsed.dataset_name
    dataset = datasets[dataset_name]
    dataset_path = basePath_data + dataset + "/" + dataset_name + ".txt"
    oracle = parsed.oracle
    print(dataset, oracle)

    seed = parsed.seed
    # set seed for reproducibility
    constants.seed = seed
    np.random.seed(seed)
    method = parsed.estimation_method
    try:
        output_path = basePath_out + dataset_name + "/"
        output_path += method + "_" + oracle
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ucc_instance = UncertainCCInstance.get_instance_from_file(dataset_path)
        print("Read graph file completed!")
        graph = ucc_instance.get_graph()
        start_time = time.time()
        if method == "jaccard":
            if "complete" in dataset_name:
                not_complete_graph = graph.copy()
                to_rem = []
                for edge in not_complete_graph.es:
                    if edge["w+_expected"]== 0.0 and edge["w-_expected"] == 1.0:
                        to_rem.append(edge.index)
                not_complete_graph.es.select(to_rem).delete()
                edges = [e.tuple for e in graph.es] # all pairs/complete graph
                jaccard_scores = not_complete_graph.similarity_jaccard(pairs=edges, loops=True)
            else:
                edges = [e.tuple for e in graph.es]
                jaccard_scores = graph.similarity_jaccard(pairs=edges, loops=True)
            w_plus = np.array(jaccard_scores)
            w_minus = 1 - w_plus
        elif method == "adamic":
            if "complete" in dataset_name:
                not_complete_graph = graph.copy()
                to_rem = []
                for edge in not_complete_graph.es:
                    if edge["w+_expected"]== 0.0 and edge["w-_expected"] == 1.0:
                        to_rem.append(edge.index)
                not_complete_graph.es.select(to_rem).delete()
                not_complete_graph = not_complete_graph.to_networkx()
                all_pairs = [(edge.source, edge.target) for edge in graph.es]
                adamic_scores = adamic_adar_index(not_complete_graph, all_pairs)
            else:
                graph_nx = graph.to_networkx()
                start_time = time.time()
                adamic_scores = adamic_adar_index(graph_nx, graph_nx.edges())
                partial_time_adamic = time.time() - start_time
            adamic_scores_dict = {}
            for s in adamic_scores:
                adamic_scores_dict[(s[0], s[1])] = s[2]
                adamic_scores_dict[(s[1], s[0])] = s[2]
            adamic_scores = []
            for edge in graph.es:
                u = edge.source
                v = edge.target
                adamic_uv = adamic_scores_dict[(u,v)]
                adamic_scores.append(adamic_uv)
            w_plus = np.array(adamic_scores)
            w_minus = 1 - w_plus
        elif method == "true":
            # clustering with true weights on the edges
            w_plus = ucc_instance.get_w_plus()
            w_minus = ucc_instance.get_w_minus()
        #assert(all(w >= 0.0 and w <= 1.0 for w in w_plus))
        if method == "adamic":
            start_time = time.time()
        delta_pivot = 1 # default value in all experiments
        num_trials_pivot = math.ceil(math.log(ucc_instance.get_number_nodes(), 1 + delta_pivot))
        A = None
        if oracle == "charikar":
            A = util.build_A(ucc_instance.get_number_nodes())
        n_runs = constants.n_runs_per_bandit
        for run_id in range(n_runs):
            if oracle == "pivot":
                clustering = util.best_pivot_run(ucc_instance, w_plus, w_minus, num_trials_pivot)
            else:
                clustering = util.best_charikar_run(ucc_instance, w_plus, w_minus, A, num_trials_pivot)

            losse_expected = ucc_instance.analytically_expected_loss(clustering, ucc_instance.get_w_plus(), ucc_instance.get_w_minus())
            
            diff_plus =  w_plus - ucc_instance.get_w_plus()
            diff_minus = w_minus - ucc_instance.get_w_minus()
            # relative L2 error
            errors = math.sqrt((np.sum(diff_plus * diff_plus) + np.sum(diff_minus * diff_minus))/(np.sum(ucc_instance.get_w_plus() * ucc_instance.get_w_plus()) + np.sum(ucc_instance.get_w_minus() * ucc_instance.get_w_minus())))

            n_clusters, avg_cluster_size = get_clustering_info(clustering)
            if run_id == 0:
                write_mode = "w+"
            else:
                write_mode = "a"
            # write output
            expected_loss_path = output_path + "/expected_losses.txt"
            with open(expected_loss_path, write_mode) as f:
                f.write(str(losse_expected) + "\n") 

            errors_path = output_path + "/errors.txt"
            with open(errors_path, write_mode) as f:
                f.write(str(errors) + "\n") 

            # best_clustering_path = output_path + "/clustering.txt"
            # with open(best_clustering_path, write_mode) as f:
            #     for c in clustering:
            #         f.write(str(c) + "\n")
            
            n_clusters_path = output_path + "/n_clusters.txt"
            with open(n_clusters_path, write_mode) as f:
                f.write(str(n_clusters) + "\n")
            
            avg_size_clusters_path = output_path + "/avg_size_clusters.txt"
            with open(avg_size_clusters_path, write_mode) as f:
                f.write(str(avg_cluster_size) + "\n")
        
        exec_time = time.time() - start_time
        exec_time /= constants.n_runs_per_bandit
        if method == "adamic":
            exec_time += partial_time_adamic
        time_path = output_path + "/time.txt"
        with open(time_path, "w+") as f:
            f.write(str(exec_time) + "\n") 
        print("Finished")

    except Exception as e:
        print(str(e))
        trace_str = traceback.format_exc()
        print(trace_str)
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        with open(output_path + "/error_log.txt", "w+") as f:
            f.write(trace_str)

if __name__ == '__main__':
    parsed = create_parser().parse_args()
    main(parsed)