import argparse as args
import os
import time
import traceback
import sys
import constants
from algorithms.eps_greedy import *
from algorithms.thompson_sampling import *
from algorithms.clcb import *
from algorithms.clcb_pivot import *
from algorithms.cmab_pivot import *
from uncertain_cc_instance import UncertainCCInstance

import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
basePath = os.path.dirname(os.path.abspath(__file__)) + "/../" 

# basePath = os.getcwd() + "/data/"
# basePath_out = os.getcwd() + "/output/"

basePath_out = basePath + "/output/"
basePath_data = basePath + "/data/"

def get_bandit(bandit, ucc_instance, exploration_probability):
    bandit_algorithm = {
        "cc-clcb": CLCB(ucc_instance),
        "cc-clcb-m": CLCB(ucc_instance, m_variant=True),
        "global-clcb": GlobalCLCB(ucc_instance),
        "global-clcb-m": GlobalCLCB(ucc_instance, m_variant=True),
        "eg": EpsilonGreedy(ucc_instance),
        "eg-fixed": EpsilonGreedy(ucc_instance, exploration_probability=exploration_probability),
        "pe": EpsilonGreedy(ucc_instance, exploration_probability=-1),
        "cts": CTS(ucc_instance),
        "pcexp-clcb": PCEXPCLCB(ucc_instance)
    }
    return bandit_algorithm[bandit]

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
    parser.add_argument('-d', '--dataset_name', help="Name of the dataset", type=str, required=True)
    parser.add_argument('-b', '--bandit', help="Bandit algorithm", choices=["cc-clcb","cc-clcb-m", "global-clcb", "global-clcb-m", "eg", "eg-fixed", "pe", "cts", "pcexp-clcb"], default="eg")
    parser.add_argument("-eps", "--exploration_probability", help="exploration probability for epsilon-greedy CMAB algorithm", type=float, default=constants.default_exploration_probability)
    parser.add_argument("-o", "--oracle", help="Oracle to use in CMAB framework", choices=["pivot", "charikar"], default="pivot")
    parser.add_argument("-T", "--timesteps", help="Number of timesteps to run the selected bandit algorithm", type=int, default=constants.default_T)
    parser.add_argument("-r", "--runs", help="Number of bandit runs", type=int, default=constants.n_runs_per_bandit)
    parser.add_argument("-s", "--seed", help="Seed value for random number generators", type=int, default=constants.seed)
    return parser

def main(parsed):
    dataset_name = parsed.dataset_name
    dataset = datasets[dataset_name]
    dataset_path = basePath_data + dataset + "/" + dataset_name + ".txt"
    bandit = parsed.bandit
    oracle = parsed.oracle
    T = parsed.timesteps
    bandit_runs = parsed.runs
    constants.n_runs_per_bandit = bandit_runs
    exploration_probability = parsed.exploration_probability
    print(dataset, bandit, oracle, T, bandit_runs)

    seed = parsed.seed
    # set seed for reproducibility
    constants.seed = seed
    np.random.seed(seed)

    try:
        output_path = basePath_out + dataset_name + "/"
        output_path += bandit 
        if bandit == "eg-fixed":
            output_path += str(exploration_probability)
        output_path += "_" + oracle
        output_path += "_T" + str(T)
        output_path += "_r" + str(bandit_runs)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ucc_instance = UncertainCCInstance.get_instance_from_file(dataset_path)
        print("Read graph file completed!")
        cmab = []
        for _ in range(constants.n_runs_per_bandit):
            cmab.append(get_bandit(bandit, ucc_instance, exploration_probability))
        n_processes = min(constants.n_runs_per_bandit, mp.cpu_count())
        print("Num processes created = " + str(n_processes))
        start_time = time.time()
        delta_pivot = 1 # default value in all experiments
        num_trials_pivot = math.ceil(math.log(ucc_instance.get_number_nodes(), 1 + delta_pivot))
        A = None
        if oracle == "pivot":
            optimal_clustering = util.best_pivot_run(ucc_instance, ucc_instance.get_w_plus(), ucc_instance.get_w_minus(), num_trials_pivot)
        else:
            A = util.build_A(ucc_instance.get_number_nodes())
            optimal_clustering = util.best_charikar_run(ucc_instance, ucc_instance.get_w_plus(), ucc_instance.get_w_minus(), A, num_trials_pivot)
        
        pool = mp.Pool(n_processes)
        seeds = np.random.randint(low=0, high=10000, size=constants.n_runs_per_bandit).tolist()
        results = [pool.apply_async(cmab_i.run, args=(T, oracle, 1, True, seeds[i], optimal_clustering, A, output_path + "/completion_log" + str(i) + ".txt", i)) for i, cmab_i in enumerate(cmab)]
        for r in results:
            r.wait()
        results = [r.get() for r in results]
        pool.close()

        exec_time = time.time() - start_time
        print("Total execution time = " + str(exec_time))
        #print(results)
        exec_time /= constants.n_runs_per_bandit

        time_path = output_path + "/time.txt"
        with open(time_path, "w+") as f:
            f.write(str(exec_time) + "\n") 

        best_clustering = None
        best_loss = float("inf")
        for run_id, result in enumerate(results):
            clustering, loss = result[0], result[1]
            if loss < best_loss:
                best_loss = loss
                best_clustering = clustering
            
            if run_id == 0:
                write_mode = "w+"
            else:
                write_mode = "a"
            
            losses_expected = result[5].tolist()
            n_clusters = result[7].tolist()
            avg_cluster_size = result[8].tolist()
            errors = result[6].tolist()
            losses_expected_notcumulative = result[9].tolist()

            avg_expected_losses_path = output_path + "/avg_expected_losses.txt"
            write_to_file(avg_expected_losses_path, write_mode, losses_expected, 2)

            errors_path = output_path + "/errors.txt"
            write_to_file(errors_path, write_mode, errors, 4)

            n_clusters_path = output_path + "/n_clusters.txt"
            write_to_file(n_clusters_path, write_mode, n_clusters, 2)

            avg_size_clusters_path = output_path + "/avg_size_clusters.txt"
            write_to_file(avg_size_clusters_path, write_mode, avg_cluster_size, 2)

            expected_losses_path = output_path + "/expected_losses.txt"
            write_to_file(expected_losses_path, write_mode, losses_expected_notcumulative, 2)

        # best_clustering_path = output_path + "/clustering.txt"
        # with open(best_clustering_path, "w+") as f:
        #     for c in best_clustering:
        #         f.write(str(c) + "\n")

        # best_loss_path = output_path + "/cost.txt"
        # with open(best_loss_path, "w+") as f:
        #     f.write(str(best_loss))

    except Exception as e:
        print(str(e))
        trace_str = traceback.format_exc()
        print(trace_str)
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        with open(output_path + "/error_log.txt", "w+") as f:
            f.write(trace_str)

def write_to_file(file_path, write_mode, data, digits):
    with open(file_path, write_mode) as f:
        line = ""
        for i, v in enumerate(data):
            v = round(v, digits)
            line += str(v) 
            if i != len(data) - 1:
                line += ";"
        line += "\n"
        f.write(line) 

if __name__ == '__main__':
    parsed = create_parser().parse_args()
    main(parsed)